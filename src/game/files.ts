import { isTauri } from '@tauri-apps/api/core';

export async function saveGameFile(content: string, filename: string): Promise<boolean> {
  if (isTauri()) {
    const [{ save }, { writeTextFile }] = await Promise.all([
      import('@tauri-apps/plugin-dialog'),
      import('@tauri-apps/plugin-fs'),
    ]);
    const path = await save({
      title: 'ツォルキンの対局を保存',
      defaultPath: filename,
      filters: [{ name: 'ツォルキンの対局', extensions: ['json'] }],
    });
    if (path === null) return false;
    await writeTextFile(path, content);
    return true;
  }

  const url = URL.createObjectURL(new Blob([content], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.hidden = true;
  document.body.append(link);
  try {
    link.click();
  } finally {
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }
  return true;
}
