"""
Privacy-preserving face anonymization with deblur-resistant heavy pixelation
"""
import cv2
import numpy as np
import random


def apply_privacy_pixelation(image, bbox):
    """
    Apply heavy pixelation with elliptical mask for natural face coverage.
    
    Uses ellipse instead of rectangle to match face shape and cover
    curved areas like jaw, cheeks, and forehead properly.
    
    Args:
        image: Input frame (numpy array)
        bbox: Tuple (x, y, w, h) defining the face region
    
    Returns:
        Modified image with heavily pixelated elliptical region
    """
    x, y, w, h = bbox
    
    # Ensure bbox is within image bounds
    img_h, img_w = image.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    
    if w <= 0 or h <= 0:
        return image
    
    # Extract face region
    region = image[y:y+h, x:x+w].copy()
    
    # Heavy downsampling: reduce to 6-10 pixels (randomized per-frame)
    target_size = random.randint(6, 10)
    
    # Calculate small dimensions maintaining aspect ratio
    if w > h:
        small_w = target_size
        small_h = max(1, int(target_size * h / w))
    else:
        small_h = target_size
        small_w = max(1, int(target_size * w / h))
    
    # Step 1: Downsample drastically (loses all facial features)
    tiny = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_AREA)
    
    # Step 2: Heavy color quantization (reduces to ~8 colors per channel)
    tiny = (tiny // 32) * 32
    
    # Step 3: Add random noise (defeats pattern recognition)
    noise = np.random.randint(-15, 16, tiny.shape, dtype=np.int16)
    tiny = np.clip(tiny.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Step 4: Random color shift per frame (prevents temporal correlation attacks)
    shift = random.randint(-10, 10)
    tiny = np.clip(tiny.astype(np.int16) + shift, 0, 255).astype(np.uint8)
    
    # Step 5: Upscale with nearest neighbor (crisp pixel blocks)
    pixelated = cv2.resize(tiny, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Step 6: Smooth edges slightly
    pixelated = cv2.GaussianBlur(pixelated, (3, 3), 0)
    
    # Create elliptical mask for natural face shape
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (w // 2, h // 2)  # Full ellipse covering the bbox
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    
    # Feather the edges for smooth blending
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
    
    # Blend pixelated region with original using elliptical mask
    blended = (pixelated.astype(np.float32) * mask_3ch + 
               region.astype(np.float32) * (1 - mask_3ch)).astype(np.uint8)
    
    # Apply back to image
    image[y:y+h, x:x+w] = blended
    
    return image


def get_face_embedding(image, bbox):
    """
    Extract a face embedding for recognition/comparison.
    
    Args:
        image: Input frame
        bbox: Tuple (x, y, w, h) for the face
    
    Returns:
        Face embedding vector or None if failed
    """
    x, y, w, h = bbox
    
    # Ensure bbox is valid
    img_h, img_w = image.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    
    if w < 50 or h < 50:
        return None
    
    try:
        # Extract and resize face for consistency
        face = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (128, 128))
        
        # Convert to grayscale for recognition
        if len(face_resized.shape) == 3:
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_resized
        
        return face_gray.flatten()  # Simple embedding
        
    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        return None


def compare_faces(embedding1, embedding2, threshold=0.6):
    """
    Compare two face embeddings using normalized correlation.
    
    Args:
        embedding1: First face embedding
        embedding2: Second face embedding
        threshold: Similarity threshold (0-1, higher = stricter)
    
    Returns:
        True if faces match, False otherwise
    """
    if embedding1 is None or embedding2 is None:
        return False
    
    try:
        # Normalize embeddings
        e1 = embedding1.astype(np.float32) / 255.0
        e2 = embedding2.astype(np.float32) / 255.0
        
        # Compute correlation coefficient
        e1_norm = e1 - np.mean(e1)
        e2_norm = e2 - np.mean(e2)
        
        correlation = np.dot(e1_norm, e2_norm) / (np.linalg.norm(e1_norm) * np.linalg.norm(e2_norm) + 1e-8)
        
        return correlation > threshold
        
    except Exception as e:
        print(f"Error comparing faces: {e}")
        return False


def apply_text_blur(image, bbox):
    """
    Apply blur to text regions.
    
    Uses a rectangular blur with feathered edges for text.
    
    Args:
        image: Input frame (numpy array)
        bbox: Tuple (x, y, w, h) defining text region
    
    Returns:
        Modified image with blurred text region
    """
    try:
        img_h, img_w = image.shape[:2]
        
        # Extract coordinates
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Add padding around text
        pad = 8
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(w + 2 * pad, img_w - x)
        h = min(h + 2 * pad, img_h - y)
        
        if w <= 5 or h <= 5:
            return image
        
        # Extract text region
        region = image[y:y+h, x:x+w].copy()
        
        # Apply heavy pixelation
        target_size = 6
        if w > h:
            small_w = max(2, target_size)
            small_h = max(1, int(target_size * h / w))
        else:
            small_h = max(2, target_size)
            small_w = max(1, int(target_size * w / h))
        
        tiny = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_AREA)
        tiny = (tiny // 32) * 32  # Color quantization
        
        # Add noise
        noise = np.random.randint(-15, 16, tiny.shape, dtype=np.int16)
        tiny = np.clip(tiny.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Upscale
        pixelated = cv2.resize(tiny, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Light blur to smooth edges
        pixelated = cv2.GaussianBlur(pixelated, (5, 5), 0)
        
        # Apply back
        image[y:y+h, x:x+w] = pixelated
        
        return image
        
    except Exception as e:
        print(f"Error blurring text: {e}")
        return image
