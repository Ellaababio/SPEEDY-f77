
import scipy.sparse as spa
import numpy as np

def test_slicing():
    # Create a simple COO matrix
    row = np.array([0, 1, 2, 0])
    col = np.array([0, 1, 2, 2])
    data = np.array([1, 2, 3, 4])
    H_coo = spa.coo_matrix((data, (row, col)), shape=(3, 3))
    
    print("Testing COO slicing (expect failure)...")
    try:
        # accessing row 0
        indices = H_coo[0, :].indices
        print("COO slicing SUCCEEDED (Unexpected!)")
    except TypeError as e:
        print(f"COO slicing failed as expected: {e}")
    except Exception as e:
        print(f"COO slicing failed with other error: {e}")

    print("\nTesting CSR slicing (expect success)...")
    try:
        H_csr = H_coo.tocsr()
        indices = H_csr[0, :].indices
        print(f"CSR slicing SUCCEEDED. Indices for row 0: {indices}")
    except Exception as e:
        print(f"CSR slicing FAILED: {e}")

if __name__ == "__main__":
    test_slicing()
