#!/usr/bin/env python3
"""
verify_outputs.py - Python Image Verification and MD5 Validator for Lab 2
Replaces complex in-C++ MD5 hashing and pixel difference loops with fast, robust Python checks.
"""

import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(SCRIPT_DIR)
COMMON_PYTHON_DIR = os.path.abspath(os.path.join(LAB_DIR, "..", "common", "python"))
if COMMON_PYTHON_DIR not in sys.path:
    sys.path.insert(0, COMMON_PYTHON_DIR)

from verify_utils import verify_image_files

def main():
    parser = argparse.ArgumentParser(description="Verify CPU and GPU generated image outputs.")
    parser.add_argument("cpu_img", nargs="?", default=os.path.join(LAB_DIR, "output_cpu.png"), help="Path to CPU output PNG")
    parser.add_argument("gpu_img", nargs="?", default=os.path.join(LAB_DIR, "output_gpu.png"), help="Path to GPU output PNG")
    args = parser.parse_args()

    print("======================================================================")
    print(" Lab 2: Python Bit-Exact Image Verification")
    print("======================================================================")
    print(f" CPU Output Image : {args.cpu_img}")
    print(f" GPU Output Image : {args.gpu_img}")
    print("----------------------------------------------------------------------")

    res = verify_image_files(args.cpu_img, args.gpu_img)
    if res.get("error"):
        print(f"[-] Verification Error: {res['error']}")
        sys.exit(1)

    print(f" CPU Image MD5    : {res['md5_a']} ({res['size_a']} bytes)")
    print(f" GPU Image MD5    : {res['md5_b']} ({res['size_b']} bytes)")
    print(f" Byte Mismatches  : {res['mismatches']}")
    print("----------------------------------------------------------------------")
    if res["matched"]:
        print(" [RESULT] : SUCCESS / PASS (100% Bit-for-bit identical outputs)")
        print("======================================================================")
        sys.exit(0)
    else:
        print(" [RESULT] : FAILURE / MISMATCH (Outputs differ)")
        print("======================================================================")
        sys.exit(1)

if __name__ == "__main__":
    main()

