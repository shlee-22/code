import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

# src를 import 경로에 추가
PROJECT_ROOT = Path("D:/상훈/논문 작성/연구논문/신척수/머신러닝/src").resolve().parents[0]
sys.path.append(str(PROJECT_ROOT / "src"))

from pipeline import run_pipeline  # noqa: E402


def main():
    # Tkinter 루트 생성
    root = tk.Tk()
    root.title("Movement Classification - Data File Selector")

    # 창 크기 & 위치 살짝 조정
    root.geometry("420x150+200+200")
    root.resizable(False, False)

    # 선택된 파일 경로를 보여줄 StringVar
    selected_path = tk.StringVar()
    selected_path.set("No file selected")

    def on_browse():
        # 윈도우 파일 선택창 열기 (엑셀 파일 필터링)
        file_path = filedialog.askopenfilename(
            title="Select Excel data file",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            selected_path.set(file_path)

    def on_run():
        file_path = selected_path.get()
        if not file_path or file_path == "No file selected":
            messagebox.showwarning("Warning", "Please select a data file first.")
            return

        # GUI 창 닫고 파이프라인 실행
        root.destroy()
        run_pipeline(file_path)

    # 라벨
    label = tk.Label(root, text="Select movement data Excel file:", anchor="w")
    label.pack(fill="x", padx=10, pady=(15, 5))

    # 경로 표시 라벨
    path_label = tk.Label(root, textvariable=selected_path, anchor="w", fg="gray")
    path_label.pack(fill="x", padx=10)

    # 버튼 프레임
    btn_frame = tk.Frame(root)
    btn_frame.pack(fill="x", padx=10, pady=15)

    browse_btn = tk.Button(btn_frame, text="Browse", command=on_browse, width=12)
    browse_btn.pack(side="left")

    run_btn = tk.Button(btn_frame, text="Run Analysis", command=on_run, width=12)
    run_btn.pack(side="right")

    root.mainloop()


if __name__ == "__main__":
    main()
