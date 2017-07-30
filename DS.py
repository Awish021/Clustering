class Crop(object):
    data = []
    def __init__(self,probe_fk,match_fk,product_name,brand_name,manufacturer_name,category_name,product_fk,mb_pk,scene_fk,probe_match_fk,bay_number,sku_sequence_number,facing_sequence_number,front_facing,height_mm,shelf_number,stacking_layer,url,bburl):
        self.probe_fk=probe_fk
        self.match_fk=match_fk
        self.product_name=product_name
        self.brand_name = brand_name
        self.manufacturer_name = manufacturer_name
        self.category_name = category_name
        self.product_fk = product_fk
        self.mb_pk = mb_pk
        self.scene_fk = scene_fk
        self.probe_match_fk = probe_match_fk
        self.bay_number = bay_number
        self.sku_sequence_number = sku_sequence_number
        self.facing_sequence_number = facing_sequence_number
        self.front_facing = front_facing
        self.height_mm = height_mm
        self.shelf_number = shelf_number
        self.stacking_layer = stacking_layer
        self.url=url
        self.bburl=bburl

    def toString(self):
        return ("Probe_fk: " + str(self.probe_fk) + '\n' + "match_fk: " + str(
            self.match_fk) + '\n' + "product_name: " + self.product_name + '\n' + "brand_name: " + self.brand_name + '\n' + "manufacturer_name: " + self.manufacturer_name + '\n' + "category_name: " + self.category_name + '\n' + "product_fk: " + str(
            self.product_fk) + '\n' + "mb_pk: " + str(self.mb_pk) + '\n' + "scene_fk: " + str(
            self.scene_fk) + '\n' + "probe_match_fk: " + str(self.probe_match_fk) + '\n' + "bay_number: " + str(
            self.bay_number) + '\n' + "sku_sequence_number: " + str(
            self.sku_sequence_number) + '\n' + "facing_sequence_number: " + str(
            self.facing_sequence_number) + '\n' + "front_facing: " + self.front_facing + '\n' + "height_mm: " + str(
            self.height_mm) + '\n' + "shelf_number: " + str(self.shelf_number) + '\n' + "stacking_layer: " + str(
            self.stacking_layer) + '\n' + "url: " + str(self.url)) + '\n' + "bburl: "+ str(self.bburl);


    def __str__(self):
        return self.toString().encode('utf-8')