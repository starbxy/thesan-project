from PIL import Image

def combine_images(image1_path, image2_path, output_path):
    # Open the two input images
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # Ensure both images have the same width
    if img1.width != img2.width:
        raise ValueError("Both images should have the same width")

    # Create a new image with the combined height
    combined_height = img1.height + img2.height
    combined_img = Image.new('RGB', (img1.width, combined_height))

    # Paste the first image at the top
    combined_img.paste(img1, (0, 0))

    # Paste the second image below the first
    combined_img.paste(img2, (0, img1.height))

    # Save the combined image to the output path
    combined_img.save(output_path)

if __name__ == "__main__":
    image1_path = "nonion.png"  # Replace with the path to your first image
    image2_path = "ion.png"  # Replace with the path to your second image
    output_path = "final.png"  # Replace with the desired output path

    combine_images(image1_path, image2_path, output_path)
