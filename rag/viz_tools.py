# utility functions
import umap
import numpy as np
# from tqdm import tqdm
from tqdm.notebook import tqdm
# ploting library
import matplotlib.pyplot as plt

# utility function to project embeddings into bidimensional space
def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings),2))
    for i, embedding in enumerate(tqdm(embeddings)):
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings

def get_projected_embeddings(_embeddings, _umap_transform):
    _projected_dataset_embeddings = project_embeddings(_embeddings, _umap_transform)
    return _projected_dataset_embeddings

def get_dataset_projected_embeddings(vectordb):
    """
    Returns a tuple with dataset projected embedings and the umap_transform
    """
    _embeddings = vectordb.get(include=['embeddings'])['embeddings']
    _umap_transform = umap.UMAP().fit(_embeddings)
    return (get_projected_embeddings(_embeddings, _umap_transform), _umap_transform)

# visualize embeddings (polar)
def plot_embeddings_query_retrieval(dataset_x, dataset_y, query_x, query_y, retrieved_x, retrieved_y):
    fig = plt.figure()
    fig.set_figwidth(12)
    fig.set_figheight(8)
    fig.suptitle('Projected Embeddings')
    fig.tight_layout()
    ax=fig.subplots(subplot_kw=dict(projection='polar'))
    ax.plot(dataset_x, dataset_y, linestyle='none', marker='.', color='gray')
    ax.plot(query_x, query_y, linestyle='none', marker='X', color='red', markersize=10)
    ax.plot(retrieved_x, retrieved_y, linestyle='none', marker='o', markerfacecolor='none', color='magenta', markersize=10)
    ax.legend(['Dataset','Query','Retrieved'])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.show()

# visualize embeddings (scater)
def plot_embeddings_query_retrieval2d(dataset_x, dataset_y, query_x, query_y, retrieved_x, retrieved_y):

    fig = plt.figure()
    fig.set_figwidth(12)
    fig.set_figheight(8)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('Projected Embeddings')
    
    plt.scatter(dataset_x, dataset_y, s=15, marker='.', color='gray')
    plt.scatter(query_x, query_y, s=150, marker='X', color='red')
    plt.scatter(retrieved_x, retrieved_y, s=50, marker='o', facecolor='none', color='magenta')
    plt.legend(['Dataset','Query','Retrieved'])

    plt.axis('on')


def plot_embeddings(mistral_dataset,
                    phi2_dataset,
                    openai_dataset,
                    bge_dataset,
                    ):
    fig = plt.figure()

    fig.set_figwidth(12)
    fig.set_figheight(8)

    gs = fig.add_gridspec(2, 2)  # , hspace=0, wspace=0)
    # (ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row', subplot_kw=dict(projection='polar'))
    (ax1, ax2), (ax3, ax4) = gs.subplots(subplot_kw=dict(projection='polar'))

    fig.suptitle('Projected Embeddings')
    fig.tight_layout()

    ax1.plot(mistral_dataset[:, 0],
             mistral_dataset[:, 1], 'b.')
    ax1.legend(['Mistral'])

    ax2.plot(phi2_dataset[:, 0],
             phi2_dataset[:, 1], 'g.')
    ax2.legend(['Phi2'])

    ax3.plot(openai_dataset[:, 0],
             openai_dataset[:, 1], 'c.')
    ax3.legend(['OpenAI'])

    ax4.plot(bge_dataset[:, 0],
             bge_dataset[:, 1], 'm.')
    ax4.legend(['all-MiniLM-L6-v2'])

    for ax in fig.get_axes():
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    plt.show()

def plot_embeddings2d(mistral_dataset,
                      phi2_dataset,
                      openai_dataset,
                      bge_dataset,
                      ):
    fig = plt.figure()

    fig.set_figwidth(12)
    fig.set_figheight(8)

    gs = fig.add_gridspec(2, 2)  # , hspace=0, wspace=0)
    # (ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row', subplot_kw=dict(projection='polar'))
    (ax1, ax2), (ax3, ax4) = gs.subplots() #subplot_kw=dict(projection='polar'))

    fig.suptitle('Projected Embeddings')
    fig.tight_layout()

    ax1.plot(mistral_dataset[:, 0],
             mistral_dataset[:, 1], 'b.')
    ax1.legend(['Mistral'])

    ax2.plot(phi2_dataset[:, 0],
             phi2_dataset[:, 1], 'g.')
    ax2.legend(['Phi2'])

    ax3.plot(openai_dataset[:, 0],
             openai_dataset[:, 1], 'c.')
    ax3.legend(['OpenAI'])

    ax4.plot(bge_dataset[:, 0],
             bge_dataset[:, 1], 'm.')
    ax4.legend(['all-MiniLM-L6-v2'])

    for ax in fig.get_axes():
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    plt.show()