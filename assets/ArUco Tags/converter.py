import cv2

# 1. Load your existing ArUco tag image
image = cv2.imread('pallet_33.png')

# 2. Define the border thickness (in pixels)
# A good rule of thumb is at least 10-20% of the image size. 
# If your tag is 200x200 pixels, a 40px border is solid.
border_size = 40 

# 3. Add the white border
bordered_image = cv2.copyMakeBorder(
    image,
    top=border_size,
    bottom=border_size,
    left=border_size,
    right=border_size,
    borderType=cv2.BORDER_CONSTANT,
    value=[255, 255, 255] # [B, G, R] color for White
)

# 4. Save the new image
cv2.imwrite('pallet_3.png', bordered_image)
print("Tag with Quiet Zone saved successfully!")