import cv2

def generate_heatmap(frame):
    # Resize for consistency
    frame = cv2.resize(frame, (500, 300))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Normalize for better contrast
    heat = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Apply color map
    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    # Blend with original
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    # Save
    cv2.imwrite("static/heatmap.jpg", overlay)
