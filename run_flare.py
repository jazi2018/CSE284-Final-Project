#!/usr/bin/env python3

import argparse
import csv
import gzip
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple


def parse_flare_ancestry_header(vcf_gz_path: str) -> Dict[str, int]:
    """
    Parse FLARE ##ANCESTRY meta-lines, returning ancestry_name -> index.
    """
    ancestry_map = {}
    with gzip.open(vcf_gz_path, "rt") as f:
        for line in f:
            if line.startswith("##ANCESTRY="):
                # Example-ish pattern: ##ANCESTRY=<ID=0,Name=CEU>
                body = line.strip()[len("##ANCESTRY=<"):-1]
                fields = {}
                for part in body.split(","):
                    key, value = part.split("=", 1)
                    fields[key] = value
                if "ID" in fields and "Name" in fields:
                    ancestry_map[fields["Name"]] = int(fields["ID"])
            elif line.startswith("#CHROM"):
                break
    return ancestry_map


def parse_flare_anc_vcf(vcf_gz_path: str, out_tsv_gz: str) -> None:
    ancestry_map = parse_flare_ancestry_header(vcf_gz_path)

    with gzip.open(vcf_gz_path, "rt") as fin, gzip.open(out_tsv_gz, "wt", newline="") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow([
            "method",
            "sample_id",
            "haplotype",
            "chrom",
            "pos",
            "marker_id",
            "pred_ancestry",
            "pred_prob"
        ])

        sample_ids = None

        for line in fin:
            if line.startswith("##"):
                continue

            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                sample_ids = fields[9:]
                continue

            fields = line.rstrip("\n").split("\t")
            chrom = fields[0]
            pos = int(fields[1])
            marker_id = fields[2]
            fmt = fields[8].split(":")
            samples = fields[9:]

            fmt_index = {key: i for i, key in enumerate(fmt)}

            has_anp1 = "ANP1" in fmt_index
            has_anp2 = "ANP2" in fmt_index

            for sample_id, sample_field in zip(sample_ids, samples):
                vals = sample_field.split(":")
                an1 = int(vals[fmt_index["AN1"]])
                an2 = int(vals[fmt_index["AN2"]])

                prob1 = ""
                prob2 = ""

                if has_anp1:
                    anp1 = vals[fmt_index["ANP1"]]
                    # ANP1 likely a comma-separated vector in ancestry index order
                    pvec1 = [float(x) for x in anp1.split(",")]
                    if 0 <= an1 < len(pvec1):
                        prob1 = pvec1[an1]

                if has_anp2:
                    anp2 = vals[fmt_index["ANP2"]]
                    pvec2 = [float(x) for x in anp2.split(",")]
                    if 0 <= an2 < len(pvec2):
                        prob2 = pvec2[an2]

                writer.writerow([
                    "flare",
                    sample_id,
                    1,
                    chrom,
                    pos,
                    marker_id,
                    an1,
                    prob1
                ])
                writer.writerow([
                    "flare",
                    sample_id,
                    2,
                    chrom,
                    pos,
                    marker_id,
                    an2,
                    prob2
                ])


def main():
    parser = argparse.ArgumentParser(description="Run FLARE and parse its output into a standardized TSV.")
    parser.add_argument("--flare-jar", required=True, help="Path to flare.jar.")
    parser.add_argument("--ref-vcf", required=True, help="Reference phased VCF.")
    parser.add_argument("--ref-panel", required=True, help="FLARE ref-panel file.")
    parser.add_argument("--study-vcf", required=True, help="Study phased VCF.")
    parser.add_argument("--map", required=True, help="PLINK-format genetic map.")
    parser.add_argument("--out-prefix", required=True, help="FLARE output prefix.")
    parser.add_argument("--parsed-out-tsv-gz", required=True, help="Standardized parsed output TSV.GZ.")
    parser.add_argument("--java-mem-gb", type=int, default=4, help="Java -Xmx memory in GB.")
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--probs", action="store_true", help="Pass probs=true to FLARE.")
    parser.add_argument("--extra-args", nargs="*", default=[], help="Additional raw FLARE key=value args.")
    args = parser.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    flare_cmd = [
        "java",
        f"-Xmx{args.java_mem_gb}g",
        "-jar",
        args.flare_jar,
        f"ref={args.ref_vcf}",
        f"ref-panel={args.ref_panel}",
        f"gt={args.study_vcf}",
        f"map={args.map}",
        f"out={args.out_prefix}",
        f"nthreads={args.nthreads}",
        f"seed={args.seed}",
        f"probs={'true' if args.probs else 'false'}",
    ] + args.extra_args

    print("Running FLARE:")
    print(" ".join(flare_cmd))

    start = time.perf_counter()
    subprocess.run(flare_cmd, check=True)
    elapsed = time.perf_counter() - start

    anc_vcf = f"{args.out_prefix}.anc.vcf.gz"
    parse_flare_anc_vcf(anc_vcf, args.parsed_out_tsv_gz)

    runtime_path = f"{args.out_prefix}.runtime.tsv"
    with open(runtime_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["method", "seconds"])
        writer.writerow(["flare", elapsed])

    print(f"Wrote parsed standardized predictions to {args.parsed_out_tsv_gz}")
    print(f"Wrote runtime to {runtime_path}")


if __name__ == "__main__":
    main()