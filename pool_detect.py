import sys
import cv2
import numpy as np
from ultralytics import YOLO

def detect_pools(input_image):
    # load the model with the best weights
    model = YOLO("weights/best.pt")

    # Perform prediction with confidence threshold 0.6 
    results = model.predict(source=input_image, conf=0.6, save=False, verbose=False)

    # read the input image
    image = cv2.imread(input_image)

    # save pool boundary coordinates to a text file
    coordinates_file = "coordinates.txt"

    with open(coordinates_file, "w") as f:
        for result in results:
            for mask in result.masks.xy:  # get polygon coordinates
                mask = np.array(mask, np.int32)  # convert them to integer
                mask = mask.reshape((-1, 1, 2))  # reshape required for drawing the red outline

                # Write coordinates to file
                np.savetxt(f, mask.reshape(-1, 2), fmt="%d", delimiter=" ")

                # Draw red outline
                cv2.polylines(image, [mask], isClosed=True, color=(0, 0, 255), thickness=2)

    # Save the output image
    output_image = "output_image.jpg"
    cv2.imwrite(output_image, image)

    print(f"Detection complete! Results saved as:\n- {coordinates_file}\n- {output_image}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detect_pools.py <input_image>")
        sys.exit(1)

    input_image = sys.argv[1]
    detect_pools(input_image)
