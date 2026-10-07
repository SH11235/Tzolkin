#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[tauri::command]
fn dispatch_game(request: String) -> Result<String, String> {
    tzolkin_core::api::dispatch_game(&request)
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![dispatch_game])
        .run(tauri::generate_context!())
        .expect("failed to run Tzolkin");
}
