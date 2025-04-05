import sys
import os

def print_environment_info():
    print("Python Executable:", sys.executable)
    print("\nPython Path:")
    for path in sys.path:
        print(f"  {path}")

def verify_imports():
    try:
        import mrd
        print("\n✓ MRD Package Imported Successfully")
        print("Package Location:", mrd.__file__)
        print("\nModule Contents:")
        print(dir(mrd))
    except ImportError as e:
        print(f"\n✗ Import Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print_environment_info()
    verify_imports()