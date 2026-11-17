import csv
import zipfile

from experiments.subsample_senders import keep, main


def test_keeps_whole_sender_histories(tmp_path):
    header = ["Timestamp", "From Bank", "Account", "To Bank", "Account", "Amount Received", "Receiving Currency",
              "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]
    rows = [[f"2022/09/01 00:{i % 60:02d}", f"{i % 7:03d}", f"8000{i % 40:05X}", "001", "800000000",
             "1", "US Dollar", "1", "US Dollar", "ACH", "0"] for i in range(400)]
    src = tmp_path / "t.csv"
    with open(src, "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    zp = tmp_path / "a.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.write(src, "T.csv")
    out = tmp_path / "o.csv"
    main(["--zip", str(zp), "--member", "T.csv", "--out", str(out), "--n", "3"])
    got = list(csv.reader(open(out)))
    assert got[0] == header
    expected = [r for r in rows if keep(r[1], r[2], 3)]
    assert got[1:] == expected and 0 < len(expected) < len(rows)
