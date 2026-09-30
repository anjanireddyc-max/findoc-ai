import os
import ast
import random
import pandas as pd
from PIL import Image


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

ANNOTATIONS_DIR = os.path.join(
    BASE_DIR,
    "annotations"
)

PATCH_DIR = os.path.join(
    BASE_DIR,
    "patches"
)


# =========================================================
# SETTINGS
# =========================================================

# Extra area around the annotated forgery
PADDING = 25

# Minimum size of the extracted patch
MIN_CROP_SIZE = 64

# Number of genuine patches per genuine document
RANDOM_GENUINE_PATCHES = 3


# =========================================================
# CREATE DIRECTORIES
# =========================================================

for split in ["train", "val", "test"]:

    os.makedirs(
        os.path.join(
            PATCH_DIR,
            split,
            "tampered"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            PATCH_DIR,
            split,
            "genuine"
        ),
        exist_ok=True
    )


# =========================================================
# CHECK FOR TAMPERED REGION
# =========================================================

def is_tampered_region(region):

    if not isinstance(region, dict):
        return False

    region_attributes = region.get(
        "region_attributes",
        {}
    )

    if not isinstance(
        region_attributes,
        dict
    ):
        return False

    original_area = region_attributes.get(
        "Original area"
    )

    return original_area == "no"


# =========================================================
# EXTRACT TAMPERED PATCH
# =========================================================

def extract_tampered_patch(
    image,
    region,
    output_path
):

    if not isinstance(
        region,
        dict
    ):
        return False

    shape_attributes = region.get(
        "shape_attributes",
        {}
    )

    if not isinstance(
        shape_attributes,
        dict
    ):
        return False

    if shape_attributes.get(
        "name"
    ) != "rect":

        return False

    try:

        x = int(
            shape_attributes["x"]
        )

        y = int(
            shape_attributes["y"]
        )

        width = int(
            shape_attributes["width"]
        )

        height = int(
            shape_attributes["height"]
        )

    except (
        KeyError,
        TypeError,
        ValueError
    ):

        return False

    if width <= 0 or height <= 0:
        return False


    # -----------------------------------------------------
    # Original bounding box
    # -----------------------------------------------------

    x1 = x
    y1 = y

    x2 = x + width
    y2 = y + height


    # -----------------------------------------------------
    # Add padding
    # -----------------------------------------------------

    x1 -= PADDING
    y1 -= PADDING

    x2 += PADDING
    y2 += PADDING


    # -----------------------------------------------------
    # Ensure minimum crop size
    # -----------------------------------------------------

    crop_width = x2 - x1
    crop_height = y2 - y1

    if crop_width < MIN_CROP_SIZE:

        extra = (
            MIN_CROP_SIZE -
            crop_width
        ) // 2

        x1 -= extra
        x2 += extra

    if crop_height < MIN_CROP_SIZE:

        extra = (
            MIN_CROP_SIZE -
            crop_height
        ) // 2

        y1 -= extra
        y2 += extra


    # -----------------------------------------------------
    # Keep coordinates inside image
    # -----------------------------------------------------

    x1 = max(
        0,
        x1
    )

    y1 = max(
        0,
        y1
    )

    x2 = min(
        image.width,
        x2
    )

    y2 = min(
        image.height,
        y2
    )


    if x2 <= x1 or y2 <= y1:
        return False


    patch = image.crop(
        (
            x1,
            y1,
            x2,
            y2
        )
    )


    patch.save(
        output_path
    )

    return True


# =========================================================
# CREATE GENUINE PATCHES
# =========================================================

def create_genuine_patches(
    image,
    image_name,
    split,
    index
):

    output_dir = os.path.join(
        PATCH_DIR,
        split,
        "genuine"
    )

    created = 0


    # Use a reasonable patch size
    patch_width = min(
        256,
        image.width
    )

    patch_height = min(
        256,
        image.height
    )


    if (
        patch_width < MIN_CROP_SIZE
        or patch_height < MIN_CROP_SIZE
    ):

        return 0


    for patch_number in range(
        RANDOM_GENUINE_PATCHES
    ):

        if image.width > patch_width:

            x = random.randint(
                0,
                image.width -
                patch_width
            )

        else:

            x = 0


        if image.height > patch_height:

            y = random.randint(
                0,
                image.height -
                patch_height
            )

        else:

            y = 0


        patch = image.crop(
            (
                x,
                y,
                x + patch_width,
                y + patch_height
            )
        )


        filename = (
            f"{index}_"
            f"{os.path.splitext(image_name)[0]}_"
            f"genuine_{patch_number}.png"
        )


        patch.save(
            os.path.join(
                output_dir,
                filename
            )
        )


        created += 1


    return created


# =========================================================
# PROCESS ONE DATASET SPLIT
# =========================================================

def process_split(split):

    csv_path = os.path.join(
        ANNOTATIONS_DIR,
        f"{split}.csv"
    )

    image_dir = os.path.join(
        DATASET_DIR,
        split
    )


    print("\n================================")
    print(
        f"PROCESSING: {split.upper()}"
    )
    print("================================")


    df = pd.read_csv(
        csv_path
    )


    print(
        f"Documents in CSV: {len(df)}"
    )


    tampered_count = 0
    genuine_count = 0
    skipped_count = 0


    # =====================================================
    # PROCESS EACH DOCUMENT
    # =====================================================

    for index, row in df.iterrows():

        image_name = str(
            row["image"]
        ).strip()


        image_path = os.path.join(
            image_dir,
            image_name
        )


        # -------------------------------------------------
        # Missing image
        # -------------------------------------------------

        if not os.path.exists(
            image_path
        ):

            print(
                f"Missing image: {image_name}"
            )

            skipped_count += 1

            continue


        # -------------------------------------------------
        # Open image
        # -------------------------------------------------

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            print(
                f"Could not open "
                f"{image_name}: {e}"
            )

            skipped_count += 1

            continue


        forged = int(
            row["forged"]
        )


        # =================================================
        # TAMPERED DOCUMENT
        # =================================================

        if forged == 1:

            annotation_value = row[
                "forgery annotations"
            ]


            # ---------------------------------------------
            # Handle missing / 0 annotations safely
            # ---------------------------------------------

            if (
                pd.isna(annotation_value)
                or str(annotation_value).strip()
                in ["0", "0.0", "", "nan"]
            ):

                print(
                    f"No annotation data: "
                    f"{image_name}"
                )

                continue


            # ---------------------------------------------
            # Convert string → dictionary
            # ---------------------------------------------

            try:

                annotation_data = ast.literal_eval(
                    str(annotation_value)
                )

            except Exception as e:

                print(
                    f"Could not parse "
                    f"{image_name}: {e}"
                )

                continue


            if not isinstance(
                annotation_data,
                dict
            ):

                print(
                    f"Invalid annotation format: "
                    f"{image_name}"
                )

                continue


            regions = annotation_data.get(
                "regions",
                []
            )


            if not isinstance(
                regions,
                list
            ):

                continue


            document_patch_count = 0


            # ---------------------------------------------
            # Extract every modified region
            # ---------------------------------------------

            for region_index, region in enumerate(
                regions
            ):

                if not is_tampered_region(
                    region
                ):

                    continue


                filename = (
                    f"{index}_"
                    f"{os.path.splitext(image_name)[0]}_"
                    f"tampered_{region_index}.png"
                )


                output_path = os.path.join(
                    PATCH_DIR,
                    split,
                    "tampered",
                    filename
                )


                success = extract_tampered_patch(
                    image,
                    region,
                    output_path
                )


                if success:

                    tampered_count += 1

                    document_patch_count += 1


            if document_patch_count == 0:

                print(
                    f"No modified regions found: "
                    f"{image_name}"
                )


        # =================================================
        # GENUINE DOCUMENT
        # =================================================

        else:

            created = create_genuine_patches(
                image,
                image_name,
                split,
                index
            )

            genuine_count += created


    # =====================================================
    # RESULTS
    # =====================================================

    print("\nResults:")

    print(
        f"Tampered patches: "
        f"{tampered_count}"
    )

    print(
        f"Genuine patches: "
        f"{genuine_count}"
    )

    print(
        f"Skipped images: "
        f"{skipped_count}"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    random.seed(42)

    process_split("train")

    process_split("val")

    process_split("test")


    print("\n================================")
    print("PATCH PREPARATION COMPLETE")
    print("================================")

    print(
        f"\nPatches saved in:\n"
        f"{PATCH_DIR}"
    )
