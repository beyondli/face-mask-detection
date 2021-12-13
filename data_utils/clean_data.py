
FACE_AREA = 40*40
WH_RATIO = 2.0
HW_RATIO = 2.0

def bbox_area(bbox):
    x_min, y_min, x_max, y_max = bbox
    #print(bbox)

    if x_max <= x_min or y_max <= y_min:
        return 0
    else:
        err_img = "{}{}{}{}".format( "x_max : ", x_max, "x_min : ", x_min)

        assert(x_max > x_min), err_img      
        assert(y_max > y_min)
        return (x_max - x_min) * (y_max - y_min)
 
def clean_bboxes(bboxes):
 
    _bboxes = []

    #import pdb; pdb.set_trace()
    '''
    for i in range(0, len(bboxes)):
        if bbox_area(bboxes[i]) > FACE_AREA :
            x_min, y_min, x_max, y_max = bboxes[i]
            _w = x_max - x_min
            _h = y_max - y_min
             
            if (_w/_h < WH_RATIO and 
                _h/_w < HW_RATIO ):
                 _bboxes.append(bboxes[i])
    '''
    if bbox_area(bboxes) > FACE_AREA :
        x_min, y_min, x_max, y_max = bboxes
        _w = x_max - x_min
        _h = y_max - y_min
             
        if (_w/_h < WH_RATIO and 
            _h/_w < HW_RATIO ):
            _bboxes = bboxes
    else:
       #print(bboxes, "==>INVALID, removed") 
       pass

    return _bboxes                
