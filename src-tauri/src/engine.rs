//! Supervises the Python science engine as a child process.
//!
//! The engine speaks one JSON object per line over stdin/stdout. A single
//! mutex guards the pipes so concurrent Tauri commands cannot interleave a
//! write with another request's read, which would desynchronise the stream.
//! stdout is read by a dedicated background thread and relayed over a
//! channel so `Engine::request` can wait for a reply with a timeout instead
//! of blocking indefinitely -- a plain `read_line` has no way to time out.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Serialize, Deserialize)]
pub struct EngineResponse {
    pub id: Option<u64>,
    pub ok: bool,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub error: Option<String>,
}

/// Bounds how long `Engine::request` waits for a reply. Generous enough for
/// the heaviest known call (repeated-seed ablation, dozens of model fits),
/// while still turning a genuinely wedged engine (e.g. a deadlocked worker
/// pool) into a visible error instead of an unbounded hang that also blocks
/// every other command behind the single request mutex.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(900);

pub struct Engine {
    child: Mutex<Option<EngineProcess>>,
    next_id: AtomicU64,
}

/// One line read from the engine's stdout, or the reason none arrived.
enum ReaderEvent {
    Line(String),
    Closed,
    Error(String),
}

struct EngineProcess {
    child: Child,
    stdin: ChildStdin,
    /// Fed by a background thread that owns the stdout pipe, so `request`
    /// can wait on it with a timeout instead of blocking on `read_line`
    /// directly (a plain blocking read has no way to time out).
    rx: mpsc::Receiver<ReaderEvent>,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            child: Mutex::new(None),
            next_id: AtomicU64::new(1),
        }
    }

    /// In development the engine runs from the project virtualenv; a packaged
    /// build ships a PyInstaller executable beside the application binary.
    fn resolve_interpreter_from(
        project_root: &std::path::Path,
        executable_dir: &std::path::Path,
    ) -> Option<(PathBuf, Vec<String>)> {
        // Tauri externalBin sidecars are placed beside the application.  The
        // second location also supports the one-folder PyInstaller layout.
        for bundled in [
            executable_dir.join("astra-engine.exe"),
            executable_dir.join("astra-engine/astra-engine.exe"),
            executable_dir.join("resources/astra-engine/astra-engine.exe"),
            // On Windows Tauri's resource directory is the executable
            // directory itself. The `..` in the release config is normalized
            // to `_up_` beside the application binary.
            executable_dir.join("_up_/engine/dist/astra-engine/astra-engine.exe"),
            // Keep the resources-prefixed form for custom bundle pipelines
            // that place the resource directory below a subdirectory.
            executable_dir.join("resources/_up_/engine/dist/astra-engine/astra-engine.exe"),
            // Keep the non-normalized form for custom bundle pipelines that
            // map the resource destination explicitly.
            executable_dir.join("resources/engine/dist/astra-engine/astra-engine.exe"),
        ] {
            if bundled.is_file() {
                return Some((bundled, Vec::new()));
            }
        }

        let venv_python = project_root.join(".venv/Scripts/python.exe");
        if venv_python.is_file() {
            return Some((venv_python, vec!["-m".into(), "astra".into()]));
        }
        None
    }

    fn resolve_interpreter() -> Option<(PathBuf, Vec<String>)> {
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .map(PathBuf::from)?;
        let executable = std::env::current_exe().ok()?;
        Self::resolve_interpreter_from(&project_root, executable.parent()?)
    }

    fn spawn() -> Result<EngineProcess, String> {
        let (program, args) = Self::resolve_interpreter().ok_or_else(|| {
            "science engine not found (no .venv and no bundled engine)".to_string()
        })?;

        // An empty argument list identifies the self-contained PyInstaller
        // sidecar.  It must run with its own import graph and resource
        // discovery; injecting the checkout's PYTHONPATH or working
        // directory here can accidentally load development code in a release
        // build (and makes the release depend on the source tree).
        let bundled = args.is_empty();

        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .map(PathBuf::from)
            .ok_or_else(|| "cannot resolve project root".to_string())?;

        let mut command = Command::new(&program);
        command
            .args(&args)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True");
        if !bundled {
            command
                .env("PYTHONPATH", project_root.join("engine"))
                .current_dir(&project_root);
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("failed to start engine at {}: {e}", program.display()))?;

        let stdin = child.stdin.take().ok_or("engine stdin unavailable")?;
        let stdout = child.stdout.take().ok_or("engine stdout unavailable")?;

        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => {
                        let _ = tx.send(ReaderEvent::Closed);
                        break;
                    }
                    Ok(_) => {
                        if tx.send(ReaderEvent::Line(line)).is_err() {
                            // Receiver dropped: request() timed out and
                            // abandoned this process. Nothing left to do.
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(ReaderEvent::Error(e.to_string()));
                        break;
                    }
                }
            }
        });

        Ok(EngineProcess { child, stdin, rx })
    }

    /// Send one request, lazily starting the engine on first use.
    ///
    /// Every failure branch drops the guarded process (`*guard = None`)
    /// before returning, not only the cleanly-closed-pipe case that already
    /// did. A dead or desynced child can be observed several ways -- a failed
    /// write to its stdin, a failed read from its stdout, or a malformed /
    /// mismatched-id response line -- and none of those are recoverable by
    /// retrying the same pipes. Leaving the guard populated with an unusable
    /// process after any of them meant every subsequent request kept failing
    /// the same way until the whole app was restarted, which is
    /// indistinguishable from "the app is crashed" even when the OS process
    /// was technically still alive.
    pub fn request(&self, method: &str, params: Value) -> Result<EngineResponse, String> {
        let mut guard = self.child.lock().map_err(|_| "engine lock poisoned")?;
        if guard.is_none() {
            *guard = Some(Self::spawn()?);
        }
        let process = guard.as_mut().expect("engine present");

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let payload = json!({ "id": id, "method": method, "params": params });

        if let Err(e) = writeln!(process.stdin, "{payload}") {
            *guard = None;
            return Err(format!("engine write failed: {e}"));
        }
        if let Err(e) = process.stdin.flush() {
            *guard = None;
            return Err(format!("engine flush failed: {e}"));
        }

        match process.rx.recv_timeout(REQUEST_TIMEOUT) {
            Ok(ReaderEvent::Line(line)) => match parse_response(&line, id) {
                Ok(response) => Ok(response),
                Err(e) => {
                    // A malformed line or an id mismatch means the stream is
                    // desynced; there is no way to resynchronise it by
                    // reading further, so treat it the same as a dead
                    // process.
                    *guard = None;
                    Err(e)
                }
            },
            Ok(ReaderEvent::Closed) => {
                *guard = None; // engine died; next request respawns it
                Err("engine closed the connection".into())
            }
            Ok(ReaderEvent::Error(e)) => {
                *guard = None;
                Err(format!("engine read failed: {e}"))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // The engine is wedged (e.g. a deadlocked worker pool). Kill
                // it rather than leaving the guard populated with a process
                // that will never answer -- that would hang every later
                // request behind this same mutex, indistinguishable from the
                // app itself being frozen.
                if let Some(mut dead) = guard.take() {
                    let _ = dead.child.kill();
                    let _ = dead.child.wait();
                }
                Err(format!(
                    "engine request timed out after {}s; the engine was restarted",
                    REQUEST_TIMEOUT.as_secs()
                ))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                *guard = None;
                Err("engine reader thread exited unexpectedly".into())
            }
        }
    }

    pub fn shutdown(&self) {
        if let Ok(mut guard) = self.child.lock() {
            if let Some(mut process) = guard.take() {
                let _ = process.child.kill();
                let _ = process.child.wait();
            }
        }
    }
}

fn parse_response(line: &str, expected_id: u64) -> Result<EngineResponse, String> {
    let response: EngineResponse =
        serde_json::from_str(line).map_err(|e| format!("malformed engine reply: {e}"))?;
    if response.id != Some(expected_id) {
        return Err(format!(
            "engine reply id mismatch: expected {expected_id}, got {:?}",
            response.id
        ));
    }
    Ok(response)
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("astra-engine-test-{unique}"));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn bundled_engine_wins_over_development_virtualenv() {
        let root = scratch();
        let exe = root.join("bin");
        fs::create_dir_all(root.join(".venv/Scripts")).unwrap();
        fs::create_dir_all(&exe).unwrap();
        fs::write(root.join(".venv/Scripts/python.exe"), b"").unwrap();
        fs::write(exe.join("astra-engine.exe"), b"").unwrap();
        let (program, args) = Engine::resolve_interpreter_from(&root, &exe).unwrap();
        assert_eq!(program, exe.join("astra-engine.exe"));
        assert!(args.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn virtualenv_is_the_development_fallback() {
        let root = scratch();
        let exe = root.join("bin");
        fs::create_dir_all(root.join(".venv/Scripts")).unwrap();
        fs::create_dir_all(&exe).unwrap();
        fs::write(root.join(".venv/Scripts/python.exe"), b"").unwrap();
        let (program, args) = Engine::resolve_interpreter_from(&root, &exe).unwrap();
        assert_eq!(program, root.join(".venv/Scripts/python.exe"));
        assert_eq!(args, vec!["-m", "astra"]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resource_folder_sidecar_is_resolved_for_installed_app() {
        let root = scratch();
        let exe = root.join("bin");
        let resource_sidecar = exe.join("resources/astra-engine/astra-engine.exe");
        fs::create_dir_all(resource_sidecar.parent().unwrap()).unwrap();
        fs::write(&resource_sidecar, b"").unwrap();

        let (program, args) = Engine::resolve_interpreter_from(&root, &exe).unwrap();
        assert_eq!(program, resource_sidecar);
        assert!(args.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn tauri_nested_resource_sidecar_is_resolved_for_installed_app() {
        let root = scratch();
        let exe = root.join("bin");
        let resource_sidecar = exe.join("resources/engine/dist/astra-engine/astra-engine.exe");
        fs::create_dir_all(resource_sidecar.parent().unwrap()).unwrap();
        fs::write(&resource_sidecar, b"").unwrap();

        let (program, args) = Engine::resolve_interpreter_from(&root, &exe).unwrap();
        assert_eq!(program, resource_sidecar);
        assert!(args.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn tauri_normalized_resource_sidecar_is_resolved_for_installed_app() {
        let root = scratch();
        let exe = root.join("bin");
        let resource_sidecar = exe.join("resources/_up_/engine/dist/astra-engine/astra-engine.exe");
        fs::create_dir_all(resource_sidecar.parent().unwrap()).unwrap();
        fs::write(&resource_sidecar, b"").unwrap();

        let (program, args) = Engine::resolve_interpreter_from(&root, &exe).unwrap();
        assert_eq!(program, resource_sidecar);
        assert!(args.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn windows_resource_sidecar_is_resolved_beside_installed_app() {
        let root = scratch();
        let exe = root.join("bin");
        let resource_sidecar = exe.join("_up_/engine/dist/astra-engine/astra-engine.exe");
        fs::create_dir_all(resource_sidecar.parent().unwrap()).unwrap();
        fs::write(&resource_sidecar, b"").unwrap();

        let (program, args) = Engine::resolve_interpreter_from(&root, &exe).unwrap();
        assert_eq!(program, resource_sidecar);
        assert!(args.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn response_framing_rejects_a_mismatched_id() {
        assert!(parse_response(r#"{"id":8,"ok":true,"result":{}}"#, 7)
            .unwrap_err()
            .contains("id mismatch"));
    }

    #[test]
    fn response_framing_accepts_one_json_line() {
        let response = parse_response(r#"{"id":7,"ok":true,"result":{"pong":true}}"#, 7).unwrap();
        assert!(response.ok);
    }
}
