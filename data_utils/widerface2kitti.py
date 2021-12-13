import scipy.io
import os
from PIL import Image, ImageDraw
 
from data_utils.clean_data import clean_bboxes
 
class widerFace2kitti():
    def __init__(self, annotation_file, widerFace_base_dir, kitti_base_dir, kitti_resize_dims, category_limit, train):
        self.annotation_file = annotation_file
        self.data = scipy.io.loadmat(self.annotation_file)
        self.file_names = self.data.get('file_list') # File Name
        self.event_list = self.data.get('event_list') # Folder Name
        self.bbox_list = self.data.get('face_bbx_list') # Bounding Boxes
        self.label_list = self.data.get('occlusion_label_list')
        self.kitti_base_dir = kitti_base_dir
        self.widerFace_base_dir = widerFace_base_dir
        self.count_mask = category_limit[0]
        self.count_no_mask = category_limit[1]
        self.kitti_resize_dims = kitti_resize_dims
        self.train = train
        self.len_dataset = len(self.file_names)

        self.count_save_image  = 0

        
        if self.train:
            try: 
                os.makedirs(self.kitti_base_dir+'/train/images',mode=0o777)
            except FileExistsError:
                print("Directory Already Exists")

            self.kitti_images = os.path.join(self.kitti_base_dir, 'train/images')
            try:
                os.makedirs(self.kitti_base_dir+ '/train/labels',mode=0o777)
            except FileExistsError:
                print("Directory Already Exists")

            self.kitti_labels = os.path.join(self.kitti_base_dir, 'train/labels')
        else:
            try:
                os.makedirs(self.kitti_base_dir+'/test/images',mode=0o777)
            except FileExistsError:
                print("Directory Already Exists")

            self.kitti_images = os.path.join(self.kitti_base_dir, 'test/images')
            try:
                os.makedirs(self.kitti_base_dir+'/test/labels',mode=0o777)
            except FileExistsError:
                print("Directory Already Exists")

            self.kitti_labels = os.path.join(self.kitti_base_dir, 'test/labels')

    def make_labels(self, image_name, category_names, bboxes):
        # Process image
        file_image = os.path.split(os.path.splitext(image_name)[0])[1]
        img = Image.open(os.path.join(self.widerFace_base_dir, image_name)).convert("RGB")
        resize_img = img.resize(self.kitti_resize_dims)
        resize_img.save(os.path.join(self.kitti_images, file_image+'.jpg'), 'JPEG')
        # Process labels
        with open(os.path.join(self.kitti_labels, file_image+ '.txt'), 'w') as label_file:
            for i in range (0, len(bboxes)):
                resized_bbox = self.resize_bbox(img=img, bbox=bboxes[i], dims=self.kitti_resize_dims)
                out_str = [category_names[i].replace(" ", "")
                           + ' ' + ' '.join(['0'] * 1)
                           + ' ' + ' '.join(['0'] * 2)
                           + ' ' + ' '.join([b for b in resized_bbox])
                           + ' ' + ' '.join(['0'] * 7)
                           + '\n']
                label_file.write(out_str[0])

    def resize_bbox(self, img, bbox, dims):
        img_w, img_h = img.size
        x_min, y_min, x_max, y_max = bbox
        ratio_w, ratio_h = img_w / dims[0], img_h / dims[1]
        new_bbox = [str(x_min / ratio_w), str(y_min / ratio_h), str(x_max / ratio_w), str(y_max / ratio_h)]
        return new_bbox

    def mat2data(self):
        count = 0
        _count_mask, _count_no_mask = 0,0
        #pick_list = ['19--Couple', '13--Interview', '16--Award_Ceremony','2--Demonstration', '22--Picnic']
        # Use following pick list for more image data
        pick_list = ['2--Demonstration', '4--Dancing', '5--Car_Accident', '15--Stock_Market', '23--Shoppers',
                      '27--Spa', '32--Worker_Laborer', '33--Running', '37--Soccer',
                      '47--Matador_Bullfighter','57--Angler', '51--Dresses', '46--Jockey',
                      '9--Press_Conference','16--Award_Ceremony', '17--Ceremony',
                      '20--Family_Group', '22--Picnic', '25--Soldier_Patrol', '31--Waiter_Waitress',
                      '49--Greeting', '38--Tennis', '43--Row_Boat', '29--Students_Schoolkids']
        for event_idx, event in enumerate(self.event_list):
            directory = event[0][0]
            if any(ele in directory for ele in pick_list):
                for im_idx, im in enumerate(self.file_names[event_idx][0]):
                    _t_count_no_mask = 0
                    im_name = im[0][0]
                    read_im_file = os.path.join(directory, im_name+'.jpg')
                    face_bbx = self.bbox_list[event_idx][0][im_idx][0]
                    category_id = self.label_list[event_idx][0][im_idx][0]
                    #  print face_bbx.shape
                    bboxes = []
                    category_names = []
                    if _count_no_mask < self.count_no_mask:
                        for i in range(face_bbx.shape[0]):
                            xmin = int(face_bbx[i][0])
                            ymin = int(face_bbx[i][1])
                            xmax = int(face_bbx[i][2]) + xmin
                            ymax = int(face_bbx[i][3]) + ymin
                            # Consider only Occlusion Free masks
                            if category_id[i][0] ==0:
                                category_name = 'No-Mask'
                                                          
                                bbox = [xmin, ymin, xmax, ymax]
                                bbox = clean_bboxes(bbox)
                                
                                if bbox:
                                    bboxes.append(bbox)
                                    category_names.append(category_name)
                                    _count_no_mask+=1
                        
                                else:
                                    #print(read_im_file, im_name, "NOT added to bboxes")
                                    pass
                        if bboxes:
                            #self.save_images(read_im_file, im_name, bboxes)
                            
                            #self.save_images(read_im_file, im_name, bboxes)
                            self.make_labels(image_name=read_im_file, category_names= category_names, bboxes=bboxes)

        print("WideFace: Total Mask Labelled:{} and No-Mask Labelled:{}".format(_count_mask, _count_no_mask))
        return _count_mask, _count_no_mask

    def test_labels(self, file_name):
        img = Image.open(os.path.join(self.kitti_images, file_name + '.jpg'))
        text_file = open(os .path.join(self.kitti_labels, file_name + '.txt'), 'r')
        features = []
        bbox = []
        category = []
        for line in text_file:
            features = line.split()
            bbox.append([float(features[4]), float(features[5]), float(features[6]), float(features[7])])
            category.append(features[0])
        print("Bounding Box", bbox)
        print("Category:", category)
        i=0
        for bb in bbox:
            draw_img = ImageDraw.Draw(img)
            shape = ((bb[0], bb[1]), (bb[2], bb[3]))
            if category[i] == 'No-Mask':
                outline_clr = "red"
            elif category[i] == 'Mask':
                outline_clr = "green"
            draw_img.rectangle(shape, fill=None, outline=outline_clr, width=4)
            i+=1

        img.show()

    def save_images(self, image_location, image_name, bboxes, debug=False):

 
        if debug:
            import pdb;pdb.set_trace()
            print(image_name, bboxes)

       #draw rect on img
        #file_image = os.path.split(os.path.splitext(image_location)[0])[1]
        file_image = image_location
        image = Image.open(os.path.join(self.widerFace_base_dir, file_image)).convert("RGB")
       
        img_bbox = ImageDraw.ImageDraw(image)
        for i in range(0, len(bboxes)):
            x_min, y_min, x_max, y_max = bboxes[i]
            shape = [(x_min, y_min), (x_max, y_max)]
            img_bbox.rectangle(shape, fill=None, outline ="green", width=5)

        self.count_save_image += 1
        image.save(os.path.join(self.kitti_images, image_name + ".jpg"), 'JPEG')

        assert(self.count_save_image < 100)


def main():
    widerFace_base_dir = '/home/nvidia/tlt-ds-face_mask_detect/dataset/WiderFace-Dataset' # Update According to dataset location
    kitti_base_dir = '/home/nvidia/tlt-ds-face_mask_detect/dataset/KITTI_960' # Update According to KITTI output dataset location
    train = True # For generating validation dataset; select False
    if train:
        annotation_file = os.path.join(widerFace_base_dir, 'wider_face_split/wider_face_train.mat')
        widerFace_base_dir = os.path.join(widerFace_base_dir, 'WIDER_train/images')
    else:
        # Modify this
        annotation_file = os.path.join(widerFace_base_dir, 'MAFA-Label-Test/LabelTestAll.mat')
        widerFace_base_dir = os.path.join(widerFace_base_dir, 'test-images\images')

    category_limit = [1000, 1000] # Mask / No-Mask Limits
    kitti_resize_dims = (960, 544) # Look at TLT model requirements
    kitti_label = widerFace2kitti(annotation_file=annotation_file, widerFace_base_dir=widerFace_base_dir,
                             kitti_base_dir=kitti_base_dir, kitti_resize_dims=kitti_resize_dims,
                             category_limit=category_limit, train=train)
    count_masks, count_no_masks = kitti_label.mat2data()
    # kitti_label.test_labels(file_name='0_Parade_Parade_0_371')
if __name__ == '__main__':
    main()
