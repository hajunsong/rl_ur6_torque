# scripts/analyze_logs.py
import os
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    csv_path = os.path.join("logs", "monitor.csv")
    df = pd.read_csv(csv_path, comment="#")
    # 모니터 CSV는 'r'(리턴), 'l'(길이), 't'(시간) 등 포함
    # 이동평균
    win = 50
    df["r_ma"] = df["r"].rolling(win, min_periods=1).mean()
    df["l_ma"] = df["l"].rolling(win, min_periods=1).mean()

    plt.figure()
    plt.plot(df["r"], label="episode return")
    plt.plot(df["r_ma"], label=f"return MA({win})")
    plt.legend(); plt.xlabel("episode"); plt.ylabel("return")
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/return_curve.png", dpi=150)

    plt.figure()
    plt.plot(df["l"], label="episode length")
    plt.plot(df["l_ma"], label=f"length MA({win})")
    plt.legend(); plt.xlabel("episode"); plt.ylabel("length")
    plt.savefig("plots/length_curve.png", dpi=150)

    print("Saved plots to plots/return_curve.png and plots/length_curve.png")
