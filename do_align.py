import os
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import pdb

def save_with_directory(path, data):
    """Ensure the directory exists and saves the numpy array or text to the specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(data, np.ndarray):
        np.save(path, data)
    else:
        with open(path, 'w') as file:
            file.write(data)

def process_matrices(matrix1, matrix2, M):
    """Calculate DTW distance, aligned matrix, and MSE between two matrices."""
    # Assuming the matrices are formatted correctly with columns as described
    formatted_matrix1 = [np.hstack((matrix1[i, :M], matrix1[i, M:2*M])) for i in range(matrix1.shape[0])]
    formatted_matrix2 = [np.hstack((matrix2[i, :M], matrix2[i, M:2*M])) for i in range(matrix2.shape[0])]
    
    distance, path = fastdtw(formatted_matrix1, formatted_matrix2, dist=lambda x, y: euclidean(x, y))
    aligned_matrix2 = np.zeros_like(matrix2)
    
    for idx1, idx2 in path:
        aligned_matrix2[idx1, :M] = matrix2[idx2, :M]
        aligned_matrix2[idx1, M:2*M] = matrix2[idx2, M:2*M]
    
    mse = np.mean((matrix1 - aligned_matrix2) ** 2)
    return distance, aligned_matrix2, mse

def main(path1, folder2, analysis_type):
    if analysis_type not in ['lips', 'hands']:
        raise ValueError("Invalid analysis type. Choose 'lips' or 'hands'.")
    
    M = 42 if analysis_type == 'lips' else 21
    suffix = f'_{analysis_type}.npy'
    folder1 = os.path.join(path1, f'MP_{analysis_type}')

    base_folder = os.path.dirname(folder2)
    output_folder = os.path.join(base_folder, f'{analysis_type}_DTW_results')
    os.makedirs(output_folder, exist_ok=True)

    files1 = sorted(os.listdir(folder1))
    files2 = sorted(os.listdir(folder2))
    file_map = {os.path.splitext(f)[0]: f for f in files2 if f.endswith(suffix)}
    
    overall_distance = 0
    overall_mse = 0
    results = []

    for file1 in files1:
        base_name = os.path.splitext(file1)[0]+"_"+analysis_type
        file2 = file_map.get(base_name)

        if file2:
            filepath1 = os.path.join(folder1, file1)
            filepath2 = os.path.join(folder2, file2)
            matrix1 = np.load(filepath1)
            matrix2 = np.load(filepath2)
            distance, aligned_matrix, mse = process_matrices(matrix1, matrix2, M)

            aligned_filename = os.path.join(output_folder, 'aligned_' + file2)
            save_with_directory(aligned_filename, aligned_matrix)

            overall_distance += distance
            overall_mse += mse
            results.append((file1, distance, mse))
            print(f"Processed {file1} and {file2}: DTW Distance = {distance}, MSE = {mse}")

    # Save summary results
    summary_content = "Filename, DTW Distance, MSE\n"
    summary_content += "\n".join(f"{res[0]}, {res[1]}, {res[2]}" for res in results)
    summary_content += f"\nOverall Total DTW Distance: {overall_distance}\n"
    summary_content += f"Overall Average MSE: {overall_mse / len(results) if results else 0}\n"
    summary_file = os.path.join(output_folder, 'summary_results.txt')
    save_with_directory(summary_file, summary_content)

# Example usage
main('/research/crissp/LPC_project/CSLM2023/MP_npy/', 'outdir/pretrained_ml_2streams/inference/filtered_resampled_inverted_both/', 'lips')  # Adjust path and analysis_type as needed
