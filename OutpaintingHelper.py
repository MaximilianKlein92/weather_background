from PIL import Image

def create_canvas(base_image, target_width, target_height):
    canvas = Image.new("RGB", (target_width, target_height), (255, 255, 255))
    x = (target_width - base_image.width) // 2
    y = (target_height - base_image.height) // 2
    canvas.paste(base_image, (x, y))

    # Maske: Weiß = ausfüllen, Schwarz = Basis behalten
    mask = Image.new("L", (target_width, target_height), 255)
    mask.paste(0, (x, y, x + base_image.width, y + base_image.height))

    return canvas, mask
