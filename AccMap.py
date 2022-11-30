from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QTextEdit
from PyQt5 import uic
import sys
from util.utils_for_gui import *
import util.utils as utils
from PyQt5 import QtGui


class AccMap(QMainWindow):
    def __init__(self):
        super(AccMap, self).__init__()
        self.count = 0
        uic.loadUi('AccMap.ui', self)
        self.load_widgets()
        self.showMaximized()  # show main window in maximum size

    def load_widgets(self):
        # create buttons
        self.open = self.findChild(QPushButton, 'open')
        self.open.clicked.connect(self.open_raw_pt)
        self.interact1 = self.findChild(QPushButton, 'interact1')
        self.interact1.clicked.connect(self.interact_raw_pt)
        self.interact2 = self.findChild(QPushButton, 'interact2')
        self.interact2.clicked.connect(self.interact_seg_pt)
        self.run1 = self.findChild(QPushButton, 'run1')
        self.run1.clicked.connect(self.run_2d_mapping)
        self.run2 = self.findChild(QPushButton, 'run2')
        self.run2.clicked.connect(self.run_segmentation)
        self.save1 = self.findChild(QPushButton, 'save1')
        self.save1.clicked.connect(self.save_2d_mapping)
        self.save2 = self.findChild(QPushButton, 'save2')
        self.save2.clicked.connect(self.save_seg_pt)
        # create display panes
        self.pane1 = self.findChild(QLabel, 'pane1')
        self.pane2 = self.findChild(QLabel, 'pane2')
        self.pane3 = self.findChild(QLabel, 'pane3')
        self.pane4 = self.findChild(QTextEdit, 'pane4')
        self.pane5 = self.findChild(QLabel, 'pane5')

    def open_raw_pt(self):
        self.raw_pt_fname, _ = QFileDialog.getOpenFileName(self, 'Open File', f'{os.getenv("HOME")}/Documents', 'PLY Files (*.ply)')
        if self.raw_pt_fname:
            # save raw pt image
            image_name = '/tmp/raw_pt_image.png'
            xyz, rgb = get_seg_coords_color(self.raw_pt_fname)
            img = makePNG(xyz, rgb)
            img = img.rotate(90)
            # img = img.crop((100, 300, 600, 1000))
            width, height = img.size
            img.resize((width // 2, height // 2)).save(image_name)
            self.pane1.setPixmap(QtGui.QPixmap(image_name))

    def interact_raw_pt(self):
        raw_pt = o3d.io.read_point_cloud(self.raw_pt_fname)
        o3d.visualization.draw_geometries([raw_pt], window_name='Interaction')

    def run_segmentation(self):
        if os.path.exists('/tmp/scans/test'):
            os.system('rm -rf /tmp/scans/test')
        ply2pth(self.raw_pt_fname)
        init()
        # model
        from model.pointgroup.pointgroup import PointGroup as Network
        from model.pointgroup.pointgroup import model_fn_decorator
        model = Network(cfg)
        use_cuda = torch.cuda.is_available()
        assert use_cuda
        model = model.cuda()
        # model_fn (criterion)
        model_fn = model_fn_decorator(test=True)
        # load model
        utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda, cfg.test_epoch, dist=False, f=cfg.pretrain)
        # evaluate
        cluster_semantic_id = test(model, model_fn, cfg.test_epoch)
        xyz, rgb = get_raw_coords_color(self.raw_pt_fname)
        rgb_out = get_semantic(f'/tmp/result/semantic_npy/{os.path.basename(self.raw_pt_fname).replace(".ply", ".npy")}')
        os.makedirs(os.path.join('/tmp/result/', 'semantic_ply'), exist_ok=True)
        self.seg_pt_fname = os.path.join('/tmp/result/', 'semantic_ply', os.path.basename(self.raw_pt_fname))
        makePlyFile(xyz, rgb_out, self.seg_pt_fname)
        # save seg pt image
        self.image_name = '/tmp/seg_pt_image.png'
        xyz, rgb = get_seg_coords_color(self.seg_pt_fname)
        img = makePNG(xyz, rgb)
        img = img.rotate(90)
        # img = img.crop((100, 300, 600, 1000))
        width, height = img.size
        img.resize((width // 2, height // 2)).save(self.image_name)
        # show seg pt image on pane
        self.pane3.setPixmap(QtGui.QPixmap(self.image_name))
        # show result
        cluster_semantic_id = cluster_semantic_id.cpu().numpy()
        semantic_id, counts = np.unique(cluster_semantic_id, return_counts=True)
        semantic_id, counts = semantic_id.tolist(), counts.tolist()
        results = 'Accessible object counts: \n'
        for each_id, count in zip(semantic_id, counts):
            results += f'{SEMANTIC_IDX2NAME[each_id]}: {count} \n'
        if self.count > 0:
            results = results + '-'*30 + '\n'
            results = results + 'Changes compared with last point cloud: \n'
            for each_id in range(1, 21):
                if each_id in semantic_id and each_id not in self.last_semantic_id:
                    results += f'{SEMANTIC_IDX2NAME[each_id]}: +{counts[semantic_id.index(each_id)]} \n'
                elif each_id not in semantic_id and each_id in self.last_semantic_id:
                    results += f'{SEMANTIC_IDX2NAME[each_id]}: -{self.last_counts[self.last_semantic_id.index(each_id)]} \n'
                elif each_id in semantic_id and each_id in self.last_semantic_id:
                    count = counts[semantic_id.index(each_id)] - self.last_counts[self.last_semantic_id.index(each_id)]
                    if count > 0:
                        results += f'{SEMANTIC_IDX2NAME[each_id]}: +{count} \n'
                    elif count < 0:
                        results += f'{SEMANTIC_IDX2NAME[each_id]}: {count} \n'
        results += '=' * 40
        self.pane4.append(results)
        self.last_semantic_id = semantic_id
        self.last_counts = counts
        self.count += 1

    def interact_seg_pt(self):
        seg_pt = o3d.io.read_point_cloud(self.seg_pt_fname)
        o3d.visualization.draw_geometries([seg_pt], window_name='Interaction')

    def run_2d_mapping(self):
        self.pane2.setPixmap(QtGui.QPixmap(self.image_name))

    def save_2d_mapping(self):
        fname, _ = QFileDialog.getSaveFileName(self, 'Save File', f'{os.getenv("HOME")}/Documents/2d_mapping.png')
        os.system(f'cp {self.image_name} {fname}')

    def save_seg_pt(self):
        fname, _ = QFileDialog.getSaveFileName(self, 'Save File', f'{os.getenv("HOME")}/Documents/3d_segmentation.ply')
        os.system(f'cp {self.seg_pt_fname} {fname}')


if __name__ == '__main__':
    app = QApplication(sys.argv)   # start App
    win = AccMap()  # open main window
    app.exec_()  # exit App
