import { useEffect, useRef } from 'react';

export function ResetDialog({
  onCancel,
  onConfirm,
  returnFocus,
}: {
  onCancel: () => void;
  onConfirm: () => void;
  returnFocus: HTMLElement | null;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const element = dialog.current;
    element?.showModal();
    return () => {
      element?.close();
      if (returnFocus?.isConnected) returnFocus.focus();
    };
  }, [returnFocus]);
  return (
    <dialog
      ref={dialog}
      className="confirm-dialog"
      aria-labelledby="reset-title"
      onCancel={onCancel}
      onKeyDown={(event) => {
        if (event.key !== 'Tab') return;
        const buttons =
          dialog.current?.querySelectorAll<HTMLButtonElement>('button:not(:disabled)');
        const first = buttons?.[0];
        const last = buttons?.[buttons.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      }}
    >
      <span className="eyebrow">新しい暦へ</span>
      <h2 id="reset-title">対局を終了しますか？</h2>
      <p>現在の対局を残す場合は、先に保存ファイルを書き出してください。</p>
      <div>
        <button className="secondary-button" autoFocus onClick={onCancel}>
          戻る
        </button>
        <button className="primary-button" onClick={onConfirm}>
          終了して新しい対局へ
        </button>
      </div>
    </dialog>
  );
}
