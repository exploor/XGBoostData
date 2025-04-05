import sys
import os

# Print detailed Python path information
print("Python Path:")
for path in sys.path:
    print(f"  {path}")

print("\nCurrent Working Directory:", os.getcwd())

# Try importing with verbose error handling
try:
    import mrd
    print("\nMRD Package Imported Successfully")
    print("Package Location:", mrd.__file__)
    
    # Additional module verification
    print("\nModule Contents:")
    print(dir(mrd))
except ImportError as e:
    print(f"\nImport Error: {e}")
    print("Import Traceback:")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"\nUnexpected Error: {e}")