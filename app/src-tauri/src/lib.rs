// ForenSim Tauri backend — Rust commands exposed to the React frontend.
use std::path::Path;
use tauri::Manager;

/// Return all image file paths in a directory (non-recursive).
#[tauri::command]
fn list_images(dir: String) -> Result<Vec<String>, String> {
    let path = Path::new(&dir);
    if !path.is_dir() {
        return Err(format!("Not a directory: {dir}"));
    }
    let entries = std::fs::read_dir(path).map_err(|e| e.to_string())?;
    let image_exts = ["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"];
    let mut images: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                .map(|ext| image_exts.contains(&ext.to_lowercase().as_str()))
                .unwrap_or(false)
        })
        .map(|e| e.path().to_string_lossy().into_owned())
        .collect();
    images.sort();
    Ok(images)
}

/// Open a native folder-picker dialog and return the selected path.
/// (Thin wrapper so the frontend doesn't need the dialog plugin directly.)
#[tauri::command]
async fn pick_folder(app: tauri::AppHandle) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    let folder = app
        .dialog()
        .file()
        .set_title("Select folder")
        .blocking_pick_folder();
    Ok(folder.map(|f| f.to_string()))
}

/// Greet command kept for reference during development.
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {name}! You've been greeted from Rust!")
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![greet, list_images, pick_folder])
        .setup(|app| {
            // In dev mode open devtools automatically.
            #[cfg(debug_assertions)]
            {
                if let Some(w) = app.get_webview_window("main") {
                    w.open_devtools();
                }
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
