import argparse

import safe as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input file with SMILES, one per line")
    parser.add_argument("--output", required=True, help="Output file for SAFE strings")
    parser.add_argument(
        "--keep-stereo",
        action="store_true",
        help="Preserve stereochemistry during SAFE encoding",
    )
    args = parser.parse_args()

    converter = sf.SAFEConverter(ignore_stereo=not args.keep_stereo)

    with open(args.input) as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    safe_list = []
    for smiles in smiles_list:
        safe_str = converter.encoder(smiles, allow_empty=True)
        safe_list.append(safe_str + "\n")

    with open(args.output, "w") as f:
        f.writelines(safe_list)


if __name__ == "__main__":
    main()
