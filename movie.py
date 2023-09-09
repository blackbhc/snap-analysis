# This script define a function to plot a gif of the given figures
# Input: figures
# Output: gif
def Movie(figures, movie_name, fps):
    """
    This function plot a gif of the given figures
    ----------
    Parameters:
        figures: list of figures
        movie_name: name of the gif
        fps: frames per second
    """
    import imageio

    images = []
    for figure in figures:
        images.append(imageio.imread(figure))
    imageio.mimsave("{}.gif".format(movie_name), images, duration=1000 / fps)
