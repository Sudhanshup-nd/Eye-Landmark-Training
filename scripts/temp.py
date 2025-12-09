from PIL import Image
from torchvision import transforms

img_path="/inwdata2a/sudhanshu/eye_multitask_training/outputs/eye_crop_overlays/0002e53f-378d-43e9-9f5f-152477561a4e/face_overlay/frame_0030_face_0_left_face_overlay_vis-True.png"
img=Image.open(img_path)


class transform_to_grayscale():
    def __call__(self, img):
        return img.convert("L")    




def transform_to_grayscale_fn(img):
    return img.convert("L")







tx_pipeline=transforms.Compose(
                                [transform_to_grayscale_fn,
                                 transforms.ToTensor()]
                               )


tx_pipeline_callable=transforms.Compose(
                                       [transform_to_grayscale(),
                                        transforms.ToTensor()]
)



bw_transformed_img=tx_pipeline_callable(img)



print(type(bw_transformed_img))


pil_img_fn=transforms.ToPILImage()
pil_img=pil_img_fn(bw_transformed_img)


pil_img.save("/inwdata2a/sudhanshu/eye_multitask_training/outputs/temp_img.png")




# bw_img=transform_to_grayscale_fn(img)

# # bw_img.save("/inwdata2a/sudhanshu/eye_multitask_training/outputs")

# bw_instance=transform_to_grayscale(img)

# bw_img_class=bw_instance(img)

# bw_img_class.save("/inwdata2a/sudhanshu/eye_multitask_training/outputs/tem_img.png")




