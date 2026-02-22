# visualization_functions.py -  Functions used for logging visualizations

# --- Setup ---

# imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import normalize as sklearn_normalize
import torch
import umap
import wandb


# --- Visualization Functions ---


# function to log image intensity histogram
def log_intensity_histogram(img_tensor, prefix, logger, step_or_epoch):

    # flatten image
    img_flat = img_tensor.detach().cpu().numpy().flatten()
    logger.experiment.log({
        f'{prefix} Intensity histogram (epoch {step_or_epoch})': wandb.Histogram(img_flat)
    })


# function to log images from multiple batches to wandb table
# batches is list of (originals, maskeds, student_preds) tuples
def log_images_batches_to_wandb_table(logger, batches, prefix, step_or_epoch, max_rows=5, log_histograms=False):

    # create table
    table = wandb.Table(columns=['Original', 'Masked', 'Student'])

    # counter for number of images logged
    global_idx = 0

    # loop through number of entries to plot and get images for plotting
    for originals, maskeds, student_preds in batches:

        batch_size = originals.shape[0]

        # stop when reach max rows to log
        for i in range(batch_size):
            if global_idx >= max_rows:
                break

            center_z = originals.shape[2] // 2
            original_img = originals[i, 0].to(torch.float32).cpu().numpy()[center_z]
            masked_img = maskeds[i, 0].to(torch.float32).cpu().numpy()[center_z]
            pred_img = student_preds[i, 0].to(torch.float32).detach().cpu().numpy()[center_z]

            # log intensity histogram for first sample in each batch
            if i == 0 and log_histograms:
                log_intensity_histogram(originals[i, 0], f'{prefix} Original', logger, step_or_epoch)
                log_intensity_histogram(maskeds[i, 0], f'{prefix} Masked', logger, step_or_epoch)
                log_intensity_histogram(student_preds[i, 0], f'{prefix} Student', logger, step_or_epoch)

            # create table
            table.add_data(
                wandb.Image(original_img, caption=f'Original {global_idx}'),
                wandb.Image(masked_img, caption=f'Masked {global_idx}'),
                wandb.Image(pred_img, caption=f'Predicted {global_idx}')
            )

            # increment counter
            global_idx += 1

        if global_idx >= max_rows:
            break
        
    # log table
    logger.experiment.log({f'{prefix} Samples (epoch {step_or_epoch})': table})


# function to log umap to wandb table
def log_embedding_umap_plot(logger, embeddings_dict, epoch, global_step, tag='Val'):

    # combine all cached image and text embeddings from the validation step
    image_embeds = torch.cat(embeddings_dict['image'], dim=0).cpu().numpy()
    text_embeds = torch.cat(embeddings_dict['text'], dim=0).cpu().numpy()
    labels = embeddings_dict['label'] # list of text description strings

    # stack all embeddings and replicate labels and types
    all_embeds = np.concatenate([image_embeds, text_embeds], axis=0) # shape: (2N, embed_dim)
    all_embeds = sklearn_normalize(all_embeds, axis=1)

    all_labels = labels + labels # same label for corresponding image and text
    embed_type = ['Image'] * len(labels) + ['Text'] * len(labels) # distinguish modality

    # reduce to 2d using umap for visualization
    reducer = umap.UMAP(n_neighbors=min(10, all_embeds.shape[0] - 1), min_dist=0.1, metric='cosine', random_state=100)
    umap_coords = reducer.fit_transform(all_embeds) # shape: (2N, 2)

    # prepare color and marker mappings for plotting
    categories = sorted(set(all_labels)) # unique stain categories
    palette = sns.color_palette('tab10', n_colors=len(categories)) # distinct color for each category
    category2color = {cat: palette[i] for i, cat in enumerate(categories)} # map from label to color
    marker_map = {'Image': 'o', 'Text': '^'} # circles for image, triangles for text

    # create umap scatterplot
    fig, ax = plt.subplots(figsize=(10, 8))
    for cat in categories:
        for typ in ['Image', 'Text']:

            # get indices for each category-modality combination
            idxs = [i for i, (l, t) in enumerate(zip(all_labels, embed_type)) if l == cat and t == typ]
            coords = umap_coords[idxs]

            # plot points with consistent color and marker
            ax.scatter(coords[:, 0], coords[:, 1],
                label=f'{cat} ({typ})',
                color=category2color[cat],
                marker=marker_map[typ],
                alpha=0.6,
                s=40)
    
    # add legend and axis labels
    ax.legend(loc='best', fontsize=7)
    ax.set_title(f'UMAP of {tag} Image/Text Embeddings (Epoch {epoch})')
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    plt.tight_layout()

    # log figure to wandb
    logger.experiment.log({f'{tag.lower()}_umap_embeddings_epoch_{epoch}': wandb.Image(fig)})
    plt.close(fig)




