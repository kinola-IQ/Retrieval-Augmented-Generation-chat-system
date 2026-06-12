from pathlib import Path

if __name__ == '__main__':
    print(Path(__file__).resolve().parents[2])
    print(Path(__file__).resolve().parents[1])
    print(Path(__file__).resolve().parents[0])