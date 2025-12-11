from setup_llm import get_embedding_vector
import os
import numpy as np
import matplotlib.pyplot as plt


def calc_proposal_embeddings(proposal_folder, IPTS_list, output_folder, embedding_model):

    '''
    in this function, we will calculate the embedding vector for each proposal and save it as a dictionary as a npz file,
    under the output_folder, we will use
    embedding_model_folder = os.path.join(output_folder, embedding_model.split("/")[-1])

    to store the results
    '''
    embedding_model_folder = os.path.join(output_folder, embedding_model.split("/")[-1])
    os.makedirs(embedding_model_folder, exist_ok=True)

    embeddings = {}
    for ipts in IPTS_list:
        file_path = os.path.join(proposal_folder, f"{ipts}.md")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        embedding = get_embedding_vector(text, embedding_model)
        embeddings[ipts] = embedding
        print(f"Calculated embedding for IPTS {ipts}")
        print(f"embedding vector (first 10 dimensions): {embedding[:10]}")

    keys = list(embeddings.keys())
    arrays = list(embeddings.values())
    np.savez(os.path.join(embedding_model_folder, 'embeddings.npz'), keys=keys, **{f'arr_{i}': arr for i, arr in enumerate(arrays)})


def analyze_proposal_similarity(embedding_model_folder):
    '''
    load the embeddings from the npz file and calculate the cosine similarity between each pair of proposals, then plot the similarity matrix as a heatmap
    return the ipts_list and similarity_matrix
    '''
    # Load the embeddings from the npz file
    data = np.load(os.path.join(embedding_model_folder, 'embeddings.npz'))
    keys = data['keys']
    arrays = [data[f'arr_{i}'] for i in range(len(keys))]

    # Calculate cosine similarity matrix
    n = len(arrays)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            u = arrays[i]
            v = arrays[j]
            similarity_matrix[i, j] = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    # Plot the similarity matrix as a pseudocolor plot
    fig, ax = plt.subplots(figsize=(12, 10))
    x = np.arange(n + 1)
    y = np.arange(n + 1)
    cax = ax.pcolormesh(x, y, similarity_matrix, cmap='coolwarm', shading='flat')
    plt.colorbar(cax)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(keys, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(keys)
    plt.title('Proposal Similarity Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(embedding_model_folder, 'similarity_matrix.png'))
    plt.close()

    # Return the ipts_list and similarity_matrix
    ipts_list = keys
    return ipts_list, similarity_matrix

