import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

DATA_ROOT = "data"

# ----------------------------------------
# Utility Functions
# ----------------------------------------

def clear_data_folder():
    """Remove entire 'data' folder before recreating the structure."""
    if os.path.exists(DATA_ROOT):
        print(f"[INFO] Removing old '{DATA_ROOT}' folder...")
        shutil.rmtree(DATA_ROOT)
    os.makedirs(DATA_ROOT, exist_ok=True)


def ensure_dirs(path_list):
    for p in path_list:
        os.makedirs(p, exist_ok=True)


def copy_files(file_list, out_dir):
    for f in file_list:
        shutil.copy(f, os.path.join(out_dir, os.path.basename(f)))


def report_missing(name, missing_list):
    if missing_list:
        print(f"[WARNING] {name} missing GT files: {missing_list}")
    else:
        print(f"[OK] {name} complete.")


# ----------------------------------------
# 1. CHASE_DB1
# ----------------------------------------

def organize_chase_db1(src):
    print("\n=== Organizing CHASE_DB1 ===")

    out = os.path.join(DATA_ROOT, "CHASE_DB1")
    ensure_dirs([
        f"{out}/train", f"{out}/train_GT",
        f"{out}/valid", f"{out}/valid_GT",
        f"{out}/test",  f"{out}/test_GT"
    ])

    img_files = sorted(glob(os.path.join(src, "*.jpg")))
    gt_files = sorted(glob(os.path.join(src, "*.png")))

    imgs = [f for f in img_files]
    gts  = [f for f in gt_files if "_1stHO" in f]

    train_gt_map = {os.path.basename(gt).replace("_1stHO.png",""): gt for gt in gts}

    print(f"Total images: {len(imgs)}, GT masks: {len(gts)}")
    train_imgs, temp_imgs = train_test_split(imgs, test_size=0.40, random_state=42)
    valid_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.50, random_state=42)

    missing_gt = []
    for image_list, out_img, out_gt in [
        (train_imgs, f"{out}/train", f"{out}/train_GT"),
        (valid_imgs, f"{out}/valid", f"{out}/valid_GT"),
        (test_imgs,  f"{out}/test",  f"{out}/test_GT")
    ]:
        for img in image_list:
            name = os.path.basename(img)
            gt = train_gt_map.get(name.replace(".jpg",""), None)
            if gt is None:
                missing_gt.append(name)
                continue
            shutil.copy(img, out_img)
            shutil.copy(gt, out_gt)

    report_missing("CHASE_DB1", missing_gt)


# ----------------------------------------
# 2. DRIVE
# ----------------------------------------

def organize_drive(src):
    print("\n=== Organizing DRIVE ===")

    out = os.path.join(DATA_ROOT, "DRIVE")
    ensure_dirs([
        f"{out}/train", f"{out}/train_GT",
        f"{out}/valid", f"{out}/valid_GT",
        f"{out}/test",  f"{out}/test_GT"
    ])

    train_imgs = sorted(glob(os.path.join(src, "training/images/*.tif")))
    train_gts  = sorted(glob(os.path.join(src, "training/1st_manual/*.gif")))

    train_gt_map = {os.path.basename(g).split('_')[0]: g for g in train_gts}

    train_imgs, valid_imgs = train_test_split(train_imgs, test_size=0.25, random_state=42)

    # TRAIN & VALID
    missing_gt = []
    for img_set, out_img, out_gt in [
        (train_imgs, f"{out}/train", f"{out}/train_GT"),
        (valid_imgs, f"{out}/valid", f"{out}/valid_GT")
    ]:
        for img in img_set:
            base = os.path.basename(img).split("_")[0]   # "21", "22", ...
            gt = train_gt_map.get(base, None)
            if gt is None:
                missing_gt.append(base)
                continue
            shutil.copy(img, out_img)
            shutil.copy(gt, out_gt)

    # TEST
    test_imgs = sorted(glob(os.path.join(src, "test/images/*.tif")))
    # test_gts = sorted(glob(os.path.join(src, "test/mask/*.tif")))

    missing_gt.extend([os.path.basename(f).split("_")[0] for f in test_imgs])
    copy_files(test_imgs, f"{out}/test")
    # copy_files(test_gts, f"{out}/test_GT")

    report_missing("DRIVE", missing_gt)


# ----------------------------------------
# 3. STARE
# ----------------------------------------

def organize_stare(src):
    print("\n=== Organizing STARE ===")

    out = os.path.join(DATA_ROOT, "STARE")
    ensure_dirs([
        f"{out}/train", f"{out}/train_GT",
        f"{out}/valid", f"{out}/valid_GT",
        f"{out}/test",  f"{out}/test_GT"
    ])

    imgs = sorted(glob(os.path.join(src, "stare-images/*.ppm")))
    gts = sorted(glob(os.path.join(src, "labels-vk/*.ppm")))
    
    train_gt_map = {os.path.basename(gt).replace(".vk",""): gt for gt in gts}

    # Just split images
    train_imgs, temp_imgs = train_test_split(imgs, test_size=0.40, random_state=42)
    valid_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.50, random_state=42)

    missing_gt = []
    for image_list, out_img, out_gt in [
        (train_imgs, f"{out}/train", f"{out}/train_GT"),
        (valid_imgs, f"{out}/valid", f"{out}/valid_GT"),
        (test_imgs,  f"{out}/test",  f"{out}/test_GT")
    ]:
        for img in image_list:
            name = os.path.basename(img)
            gt = train_gt_map.get(name, None)
            if gt is None:
                missing_gt.append(name)
                continue
            shutil.copy(img, out_img)
            shutil.copy(gt, out_gt)

    report_missing("STARE", missing_gt)


# ----------------------------------------
# 4. Lung Dataset
# ----------------------------------------

def organize_lung(src):
    print("\n=== Organizing Lung Dataset ===")

    out = os.path.join(DATA_ROOT, "Lung")
    ensure_dirs([
        f"{out}/train", f"{out}/train_GT",
        f"{out}/valid", f"{out}/valid_GT",
        f"{out}/test",  f"{out}/test_GT"
    ])

    imgs = sorted(glob(os.path.join(src, "2d_images/*.tif")))
    gts  = sorted(glob(os.path.join(src, "2d_masks/*.tif")))

    train_gt_map = {os.path.basename(m): m for m in gts}

    train_imgs, temp_imgs = train_test_split(imgs, test_size=0.40, random_state=42)
    valid_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.50, random_state=42)

    missing_gt = []
    for file_list, out_img, out_gt in [
        (train_imgs, f"{out}/train", f"{out}/train_GT"),
        (valid_imgs, f"{out}/valid", f"{out}/valid_GT"),
        (test_imgs,  f"{out}/test",  f"{out}/test_GT")
    ]:
        for img in file_list:
            name = os.path.basename(img)
            gt = train_gt_map.get(name, None)
            if gt is None:
                missing_gt.append(name)
                continue
            shutil.copy(img, out_img)
            shutil.copy(gt, out_gt)

    report_missing("Lung", missing_gt)


# ----------------------------------------
# 5. Skin Cancer (ISIC2017)
# ----------------------------------------

def organize_skin(src):
    print("\n=== Organizing Skin Cancer (ISIC2017) ===")

    out = os.path.join(DATA_ROOT, "SkinCancer")
    ensure_dirs([
        f"{out}/train", f"{out}/train_GT",
        f"{out}/valid", f"{out}/valid_GT",
        f"{out}/test",  f"{out}/test_GT"
    ])

    train_imgs = sorted(glob(os.path.join(src, "ISIC-2017_Training_Data/*.jpg")))
    train_gts  = sorted(glob(os.path.join(src, "ISIC-2017_Training_Part1_GroundTruth/*.png")))

    valid_imgs = sorted(glob(os.path.join(src, "ISIC-2017_Validation_Data/*.jpg")))
    valid_gts  = sorted(glob(os.path.join(src, "ISIC-2017_Validation_Part1_GroundTruth/*.png")))

    test_imgs = sorted(glob(os.path.join(src, "ISIC-2017_Test_v2_Data/*.jpg")))
    test_gts  = sorted(glob(os.path.join(src, "ISIC-2017_Test_v2_Part1_GroundTruth/*.png")))

    missing_list = []
    for img_list, gt_list in [
        (train_imgs, train_gts),
        (valid_imgs, valid_gts),
        (test_imgs,  test_gts)
    ]:
        img_bases = {os.path.basename(f).replace(".jpg", "") for f in img_list}
        gt_bases  = {os.path.basename(f).replace("_segmentation.png", "") for f in gt_list}
        missing_list.extend(img_bases - gt_bases)

    copy_files(train_imgs, f"{out}/train")
    copy_files(train_gts,  f"{out}/train_GT")

    copy_files(valid_imgs, f"{out}/valid")
    copy_files(valid_gts,  f"{out}/valid_GT")

    copy_files(test_imgs,  f"{out}/test")
    copy_files(test_gts,   f"{out}/test_GT")

    report_missing("Skin Cancer", missing_list)


# ----------------------------------------
# Run All
# ----------------------------------------

def run():
    clear_data_folder()
    
    organize_chase_db1("../dataset/Blood Vessel Segmentation/CHASE_DB1")
    organize_drive("../dataset/Blood Vessel Segmentation/DRIVE")
    organize_stare("../dataset/Blood Vessel Segmentation/STARE")
    organize_lung("../dataset/Lung Segmentation")
    organize_skin("../dataset/Skin Cancer Segmentation")


if __name__ == "__main__":
    run()
