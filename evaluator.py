from downstream_clustering.kmeans_based import features_kmeans, features_n_kmeans
from munkres import Munkres, print_matrix
from sklearn import metrics
import numpy as np
import os
# from EMAN2 import EMNumPy,Averagers
import stat
import shutil
from tqdm import tqdm
import multiprocessing
from functools import partial
from PIL import Image
import torch
from torchvision.utils import save_image
import pandas as pd
from cryoemdata.data_preprocess.mrc_preprocess import mrcs_resize
from models.get_transformers import filter
import cv2
import pickle
from cryoemdata.data_preprocess.mrc_preprocess import mrcs_to_int8

# from cryoemdata.e2classaverage import class_average_withali, class_average
# from EMAN2 import Transform

class Clustering_evaluator(object):
    def __init__(self, particles_path, result_path_dir, k, particles_len,particles_id=None, labels_p=None, labels_t=None,
                 features=None):
        self.result_path = result_path_dir
        self.particles_path = particles_path
        if particles_id is None:
            self.particles_id = range(particles_len)
        else:
            self.particles_id = particles_id
        self.labels_p = labels_p
        self.labels_t = labels_t
        self.k = k
        self.features = features

    def clustering(self, features,k=None):
        if k==None:
            k=self.k
        self.features = features
        # labels_predict, centers, num_class = features_kmeans(features,k=self.k)
        labels_predict, centers, num_class = features_n_kmeans(features, k=k)
        self.labels_p = labels_predict
        self.centers = centers
        return labels_predict, centers, num_class

    def acc_nmi(self, labels_t):
        self.labels_t = labels_t
        label_same = best_map(self.labels_t, self.labels_p)
        count = np.sum(self.labels_t[:] == label_same[:])
        acc = count.astype(float) / (len(self.labels_t))
        nmi = metrics.normalized_mutual_info_score(self.labels_t, label_same)
        return acc, nmi

    def save_labels(self, epoch, particles_id=None,save_path_dir_name=None):
        if particles_id is None:
            ids = self.particles_id
        else:
            ids = particles_id
        col_name = ['clustering_results']
        df = pd.DataFrame(columns=col_name, data=self.labels_p, index=ids)
        if save_path_dir_name is None:
            labels_predict_path = self.result_path + '/clustering_predict/epoch_' + str(epoch)
        else:
            labels_predict_path = os.path.join(self.result_path, save_path_dir_name, '/epoch_' + str(epoch))
        # labels_predict_path = self.result_path + '/clustering_predict/epoch_' + str(epoch)
        if not os.path.exists(labels_predict_path):
            os.makedirs(labels_predict_path)
        df.to_csv(labels_predict_path + '/clustering_predict.csv', encoding='utf-8')

        with open(labels_predict_path + '/output_tif_label.data', 'wb') as filehandle:
            pickle.dump(self.labels_p, filehandle)
        # with open(labels_predict_path + '/clustering_predict.data', 'wb') as filehandle:
        #     pickle.dump(self.labels_p, filehandle)

    # def save_particles_list(self):
    #     col_name = ['particles']
    #     df = pd.DataFrame(columns=col_name, data=self.particles_path)
    #     labels_predict_path = self.result_path + '/clustering_predict/'
    #     if not os.path.exists(labels_predict_path):
    #         os.makedirs(labels_predict_path)
    #     df.to_csv(labels_predict_path + '/particles_list.csv', encoding='utf-8', index=False)

        # with open(labels_predict_path + '/clustering_predict.data', 'wb') as filehandle:
        #     pickle.dump(self.labels_p, filehandle)

    def save_averages(self, epoch):
        my_save_averages(epoch, self.result_path, self.averages, self.original_averages)
        # averages_predict_path = self.result_path + '/clustering_predict/epoch_' + str(epoch)
        # if not os.path.exists(averages_predict_path):
        #     os.makedirs(averages_predict_path)
        # with open(averages_predict_path + '/averages.data', 'wb') as filehandle:
        #     pickle.dump(self.original_averages, filehandle)
        # if np.ndim(self.averages) == 3:
        #     self.averages=np.expand_dims(self.averages, axis=1)
        # save_image(torch.tensor(self.averages), averages_predict_path + '/averages.png')
        # for i in range(self.averages.shape[0]):
        #     Image.fromarray(self.averages[i]).save(averages_predict_path+'/'+str(i)+'.pdf')

    def update_labels(self, labels_p=None, labels_t=None):
        self.labels_p = labels_p
        self.labels_t = labels_t

    def embedding_projection(self, tb_writer, labels_t=None, title=None, epoch=None):
        if self.labels_p.shape[0] > 20000:
            if labels_t is not None:
                sub_labels = labels_t[:20000]
            else:
                sub_labels = self.labels_p[:20000]
            sub_features = self.features[:20000]
        else:
            if labels_t is not None:
                sub_labels = labels_t
            else:
                sub_labels = self.labels_p
            sub_features = self.features
        tb_writer.add_embedding(mat=sub_features, metadata=sub_labels, global_step=epoch, tag=title)

    @torch.no_grad()
    def knn_classifier(self, train_features=None, train_labels=None, test_features=None, test_labels=None,
                       num_classes=None, k=5, T=1,mask_start_index=0):
        if num_classes is None:
            num_classes = self.k
        if train_labels is None:
            train_labels = self.labels_t
        if test_labels is None:
            test_labels = self.labels_t
        if train_features is None:
            train_features = self.features
        if test_features is None:
            test_features = self.features

        train_features = torch.from_numpy(train_features).cuda()
        train_labels = torch.from_numpy(np.asarray(train_labels)).cuda()
        test_features = torch.from_numpy(test_features).cuda()
        test_labels = torch.from_numpy(np.asarray(test_labels)).cuda()

        top1, top5, total = 0.0, 0.0, 0
        train_features = train_features.t()
        num_test_images, num_chunks = test_labels.shape[0], int(test_labels.shape[0]/1000)
        imgs_per_chunk = num_test_images // num_chunks
        retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
        for idx in range(0, num_test_images, imgs_per_chunk):
            # get the features for test images
            features = test_features[
                       idx: min((idx + imgs_per_chunk), num_test_images), :
                       ]
            targets = test_labels[idx: min((idx + imgs_per_chunk), num_test_images)]
            batch_size = targets.shape[0]

            # calculate the dot product and compute top-k neighbors
            similarity = torch.mm(features, train_features)
            # similarity_np=similarity.cpu().numpy()
            # mm=torch.arange(idx, min((idx + imgs_per_chunk), num_test_images)).unsqueeze(-1)
            mask = torch.ones_like(similarity).scatter_(1, torch.arange(mask_start_index+idx, mask_start_index+min((idx + imgs_per_chunk),
                                                                                 num_test_images)).unsqueeze(-1).cuda(),
                                                        0)
            # aaaa = mask.cpu().numpy()
            similarity = similarity.masked_fill(mask == 0, 0)
            distances, indices = similarity.topk(k, largest=True, sorted=True)
            candidates = train_labels.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
            distances_transform = distances.clone().div_(T).exp_()

            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    distances_transform.view(batch_size, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)
            # predictions_np=predictions.cpu().numpy()

            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
            total += targets.size(0)
        top1 = top1 * 100.0 / total
        top5 = top5 * 100.0 / total
        return top1, top5

    def imgs_align_v2(self, val_particles_path, val_particles_ctf_path, align_batch=10000, iteration_times=3,
                      multiprocessing_for_classes=True, resize=None, low_pass=None, pixel_size=None):

        k = self.k
        phbar = tqdm(range(k))
        phbar.set_description("rotating correction")
        average_imgs_path = self.result_path + "/averages/"

        raw_imgs_path_list = np.asarray(val_particles_path)
        raw_imgs_path_list_ctf = np.asarray(val_particles_ctf_path)

        labels = self.labels_p
        # pionts_nearest_to_centers = get_points_nearest_to_centers(self.features, self.centers, self.labels_p)
        for j in phbar:
            # find children and generate averages

            children_path = raw_imgs_path_list[labels == j]
            children_path = children_path[0:10000] if children_path.shape[0] > 10000 else children_path
            children_path_ctf = raw_imgs_path_list_ctf[labels == j]
            children_path_ctf = children_path_ctf[0:10000] if children_path_ctf.shape[0] > 10000 else children_path_ctf

            children = read_imgs_from_path_list_to_em(children_path)
            children_ctf = read_imgs_from_path_list_to_em(children_path_ctf)

            # ref = EMNumPy.numpy2em(np.asarray(pickle.load(open(raw_imgs_path_list[int(pionts_nearest_to_centers[j])], 'rb'))))

            # average_ctf = class_average(children_ctf,ref=ref)
            average_ctf = class_average(children_ctf)
            # average_ctf_np = to_int8(EMNumPy.em2numpy(average_ctf[0]))

            average = class_average_withali(images=children, ptcl_info=average_ctf[1], ref=average_ctf[0],
                                            xform=Transform(), focused=None)
            # from copy import deepcopy
            # average_np_1 = deepcopy(average)

            average_np = EMNumPy.em2numpy(average).copy()
            average_np = np.expand_dims(average_np, axis=0)
            # save the averages
            if j == 0:
                averages = average_np
            else:
                averages = np.append(averages, average_np, axis=0)
        if not os.path.exists(average_imgs_path):
            os.makedirs(average_imgs_path)
        # save_image(torch.unsqueeze(torch.from_numpy(averages), 1)
        #            average_imgs_path + '/clustering_result' + str(epoch) + '.png')
        # averages=mrcs_to_int8(averages)
        self.original_averages = averages
        averages = mrcs_norm_to_0_1(averages.copy())
        averages = np.expand_dims(averages, axis=1)
        self.averages = averages
        # average = np.expand_dims(average, axis=0)
        return averages

    def imgs_align_v2_multi(self, val_particles_path, val_particles_ctf_path, align_batch=10000, iteration_times=3,
                            multiprocessing_for_classes=True, resize=None, low_pass=None, pixel_size=None):

        average_imgs_path = self.result_path + "/averages/"
        if not os.path.exists(average_imgs_path):
            os.makedirs(average_imgs_path)

        averages_list = imgs_align_multi(k=self.k, labels=self.labels_p, val_particles_path=val_particles_path,
                                         val_particles_ctf_path=val_particles_ctf_path, resize=resize)

        # save_image(torch.unsqueeze(torch.from_numpy(averages), 1)
        #            average_imgs_path + '/clustering_result' + str(epoch) + '.png')
        # averages=mrcs_to_int8(averages)
        self.original_averages = averages_list
        averages = mrcs_norm_to_0_1(averages_list.copy())
        # averages = np.expand_dims(averages, axis=1)
        self.averages = averages
        # average = np.expand_dims(average, axis=0)
        return averages

    def imgs_align(self, val_particles_path=None, align_batch=10000, iteration_times=3,
                   multiprocessing_for_classes=True, resize=None, low_pass=None, pixel_size=None):
        k = self.k
        phbar = tqdm(range(k))
        phbar.set_description("rotating correction")
        average_imgs_path = self.result_path + "/averages/"
        total_num = 0
        path_raw_to_align = []
        if val_particles_path is not None:
            raw_imgs_path_list = np.asarray(val_particles_path)
        else:
            raw_imgs_path_list = np.asarray(self.particles_path)
        labels = self.labels_p
        pionts_nearest_to_centers = get_points_nearest_to_centers(self.features, self.centers, self.labels_p)
        for j in phbar:
            # find children and generate averages
            if isinstance(raw_imgs_path_list[0], str):
                children_path = raw_imgs_path_list[labels == j]
                path_raw_to_align += children_path.tolist()

                children_iter_times = int(children_path.shape[0] / align_batch)
                if children_path.shape[0] % align_batch > 0:
                    children_iter_times += 1
                # children_path=children_path[0:10000] if children_path.shape[0]>10000 else children_path
                children = read_imgs_from_path_list(children_path)
                if low_pass is not None and low_pass['use_low_pass']:
                    low_pass_scale = low_pass['scale']
                    children = mrcs_low_pass(children, low_pass_scale, pixel_size)
                if resize is not None:
                    children = mrcs_resize(children, resize, resize)
                # ref = EMNumPy.numpy2em(np.asarray(Image.open(raw_imgs_path_list[int(pionts_nearest_to_centers[j])])))
                ref = EMNumPy.numpy2em(
                    np.asarray(pickle.load(open(raw_imgs_path_list[int(pionts_nearest_to_centers[j])], 'rb'))))
            else:
                children = raw_imgs_path_list[labels == j]
                children = children[0:10000] if children.shape[0] > 10000 else children
                if low_pass is not None and low_pass['use_low_pass']:
                    children = mrcs_low_pass(children, low_pass, pixel_size)
                if resize is not None:
                    children = mrcs_resize(children, resize, resize)
                children_iter_times = int(children.shape[0] / align_batch)
                if children.shape[0] % align_batch > 0:
                    children_iter_times += 1
                ref = EMNumPy.numpy2em(raw_imgs_path_list[int(pionts_nearest_to_centers[j])])
            # children = children[np.asarray(reliable_list)[labels == j] == True]
            num_children = children.shape[0]
            for i in range(iteration_times):
                # print("Iter ", i)
                avgr = Averagers.get("mean", {"ignore0": True})
                average_list_i = []
                aligned_list = []
                for ii in range(children_iter_times):
                    # e=(ii+1)*align_batch if (ii+1)*align_batch<num_children else num_children+1
                    if (ii + 1) * align_batch < num_children:
                        item = [children[ii * align_batch:(ii + 1) * align_batch][i] for i in range(align_batch)]
                    else:
                        item = [children[ii * align_batch:][i] for i in range(num_children % align_batch)]
                    # item = [children for i in range(num_children)]
                    func = partial(aligin_one_img, ref=ref)
                    pool = multiprocessing.Pool(15)
                    aligined_img = pool.map(func, item)
                    aligned_list.append(aligined_img)
                    pool.close()
                    pool.join()
                    for num in range(len(aligined_img)):
                        avgr.add_image(aligined_img[num])
                        # aligined_imgs[total_num + num] = EMNumPy.em2numpy(aligined_img[num])
                    ref = avgr.finish()
            # avfr_np= EMNumPy.em2numpy(avgr)
            average = EMNumPy.em2numpy(ref)
            average = np.expand_dims(average, axis=0)
            # save the averages
            if j == 0:
                averages = average.copy()
            else:
                averages = np.append(averages, average, axis=0)
            total_num += num_children
        if not os.path.exists(average_imgs_path):
            os.makedirs(average_imgs_path)
        # save_image(torch.unsqueeze(torch.from_numpy(averages), 1),
        #            average_imgs_path + '/clustering_result' + str(epoch) + '.png')
        # averages=mrcs_to_int8(averages)
        self.original_averages = averages
        averages = mrcs_norm_to_0_1(averages.copy())
        averages = np.expand_dims(averages, axis=1)
        self.averages = averages
        # average = np.expand_dims(average, axis=0)
        return averages


def imgs_align_multi(k, labels, val_particles_path, val_particles_ctf_path, resize=None):
    phbar = tqdm(range(k))
    phbar.set_description("rotating correction")
    raw_imgs_path_list = np.asarray(val_particles_path)
    raw_imgs_path_list_ctf = np.asarray(val_particles_ctf_path)
    func = partial(class_align, raw_imgs_path_list=raw_imgs_path_list, raw_imgs_path_list_ctf=raw_imgs_path_list_ctf,
                   labels=labels, resize=resize)
    pool = multiprocessing.Pool(15)
    averages_list = np.asarray(pool.map(func, phbar))
    pool.close()
    pool.join()
    # averages = mrcs_norm_to_0_1(averages_list.copy())
    return averages_list


def my_save_averages(epoch, result_path, averages, original_averages):
    averages_predict_path = result_path + '/clustering_predict/epoch_' + str(epoch)
    if not os.path.exists(averages_predict_path):
        os.makedirs(averages_predict_path)
    with open(averages_predict_path + '/averages.data', 'wb') as filehandle:
        pickle.dump(original_averages, filehandle)
    if np.ndim(averages) == 3:
        averages = np.expand_dims(averages, axis=1)
    save_image(torch.tensor(averages), averages_predict_path + '/averages.png')


def best_map(L1, L2):
    # L1 should be the labels and L2 should be the downstream_clustering number we got
    Label1 = np.unique(L1)  # 去除重复的元素，由小大大排列
    nClass1 = len(Label1)  # 标签的大小
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def get_points_nearest_to_centers(points_data, centers, labels):
    k = centers.shape[0]
    points_nearest_to_centers = np.zeros(k)
    mindist = np.full(k, np.inf)
    for i in range(points_data.shape[0]):
        class_id = labels[i]
        dist = np.linalg.norm(points_data[i] - centers[class_id])
        if dist < mindist[class_id]:
            points_nearest_to_centers[class_id] = i
            mindist[class_id] = dist
    return points_nearest_to_centers


def delete_file(filePath):
    if os.path.exists(filePath):
        for fileList in os.walk(filePath):
            for name in fileList[2]:
                os.chmod(os.path.join(fileList[0], name), stat.S_IWRITE)
                os.remove(os.path.join(fileList[0], name))
        shutil.rmtree(filePath)
        return "delete ok"
    else:
        return "no filepath"


def class_align(j, raw_imgs_path_list, raw_imgs_path_list_ctf, labels, resize):
    # find children and generate averages

    children_path = raw_imgs_path_list[labels == j]
    children_path = children_path[0:10000] if children_path.shape[0] > 10000 else children_path
    # children_path_ctf = raw_imgs_path_list_ctf[labels == j]
    children_path_ctf = raw_imgs_path_list[labels == j]
    children_path_ctf = children_path_ctf[0:10000] if children_path_ctf.shape[0] > 10000 else children_path_ctf

    children = read_imgs_from_path_list_to_em(children_path, resize=resize)

    children_ctf = read_imgs_from_path_list_to_em(children_path_ctf, resize=resize)

    # ref = EMNumPy.numpy2em(np.asarray(pickle.load(open(raw_imgs_path_list[int(pionts_nearest_to_centers[j])], 'rb'))))

    # average_ctf = class_average(children_ctf,ref=ref)
    average_ctf = class_average(children_ctf)
    # average_ctf_np = to_int8(EMNumPy.em2numpy(average_ctf[0]))

    average = class_average_withali(images=children, ptcl_info=average_ctf[1], ref=average_ctf[0],
                                    xform=Transform(), focused=None)
    # from copy import deepcopy
    # average_np_1 = deepcopy(average)

    average_np = EMNumPy.em2numpy(average).copy()
    average_np = np.expand_dims(average_np, axis=0)

    return average_np


def aligin_one_img(im, ref):
    im = EMNumPy.numpy2em(im)
    # im.process_inplace("normalize.edgemean")
    # if im["nx"]!=nx or im["ny"]!=ny :
    # 	im=im.get_clip(Region(old_div(-(nx-im["nx"]),2),old_div(-(ny-im["ny"]),2),nx,ny))
    # im.write_image("result/seq.mrc",-1)
    # ima=im.align("translational",ref0,{"nozero":1,"maxshift":old_div(ref0["nx"],4.0)},"ccc",{})
    ima = im.align("rotate_translate_tree", ref)
    # ima = EMNumPy.em2numpy(ima)
    return ima


def read_imgs_from_path_list(paths):
    imgs = []
    for path in paths:
        # imgs.append(np.asarray(Image.open(path)))
        imgs.append(np.asarray(pickle.load(open(path, 'rb'))))
    return np.asarray(imgs)


def read_imgs_from_path_list_to_em(paths, resize):
    imgs = []
    for path in paths:
        if resize is not None:
            img = pickle.load(open(path, 'rb')).resize((resize, resize), Image.Resampling.BICUBIC)
        else:
            img = pickle.load(open(path, 'rb'))
        # imgs.append(np.asarray(Image.open(path)))
        imgs.append(EMNumPy.numpy2em(np.asarray(img)))
    return imgs


def mrcs_norm_to_0_1(mrcs):
    if torch.is_tensor(mrcs):
        for i in range(mrcs.shape[0]):
            # m=torch.max(mrcs[i])
            mrcs[i] = (mrcs[i] - torch.min(mrcs[i])) / (torch.max(mrcs[i]) - torch.min(mrcs[i]))
    else:
        for i in range(mrcs.shape[0]):
            mrcs[i] = (mrcs[i] - np.min(mrcs[i])) / ((np.max(mrcs[i]) - np.min(mrcs[i])))
    return mrcs


def mrcs_low_pass(mrcs, cutoff_A, pixelsize):
    img_width = mrcs.shape[1]
    cutoff_pixel = int(pixelsize * img_width / cutoff_A)
    mask = filter(img_width, cutoff_pixel, filter='gaussian', type='lp')
    for i in range(mrcs.shape[0]):
        img = mrcs[i]
        f = cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        f_shifted = np.fft.fftshift(f)

        # f_filtered = f_shifted * mask
        f_complex = f_shifted[:, :, 0] * 1j + f_shifted[:, :, 1]
        f_filtered = mask * f_complex
        f_filtered_shifted = np.fft.fftshift(f_filtered)
        inv_img = np.fft.ifft2(f_filtered_shifted)
        filtered_img = np.real(inv_img)
        mrcs[i] = filtered_img
    return mrcs


def get_averages_with_labels():
    # from e2classaverage import class_average_withali,class_average
    # from EMAN2 import Transform
    # label_path='/yanyang2/dataset/clustering/cs_simulated_particles/emd_6840/withlabels/valset_snr01_5/output_tif_label.data'
    # label_path='/yanyang2/projects/results/particle_clustering_inference/emd6840_valset/1220_snr01/test1/clustering_predict/epoch_291/output_tif_label.data'
    label_path = '/yanyang2/dataset/clustering/cs_simulated_particles/j1461/withlabels/realdata_1228_ns05/cs_predict_label.data'
    # imgs_path='/yanyang2/dataset/clustering/cs_simulated_particles/emd_6840/withlabels/valset_snr01_5/output_tif_path.data'
    imgs_path = '/yanyang2/dataset/clustering/cs_simulated_particles/j1461/withlabels/realdata_1228_ns05/output_tif_path.data'
    # imgs_path_ctf='/yanyang2/dataset/clustering/cs_simulated_particles/emd_6840/withlabels/valset_snr01_5/output_ctf_tif_path.data'
    imgs_path_ctf = '/yanyang2/dataset/clustering/cs_simulated_particles/j1461/withlabels/realdata_1228_ns05/output_ctf_tif_path.data'
    k = 9
    save_path = '/yanyang2/dataset/test/average_test/'
    with open(label_path, 'rb') as filehandle:
        label_path_list = np.asarray(pickle.load(filehandle))
    with open(imgs_path, 'rb') as filehandle:
        imgs_path_list = np.asarray(pickle.load(filehandle))
    with open(imgs_path_ctf, 'rb') as filehandle:
        imgs_path_ctf_list = np.asarray(pickle.load(filehandle))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    averages_list = imgs_align_multi(k=k, labels=label_path_list, val_particles_path=imgs_path_list,
                                     val_particles_ctf_path=imgs_path_ctf_list, resize=128)
    averages = mrcs_norm_to_0_1(averages_list.copy())
    my_save_averages(0, save_path, averages, averages_list)

    # em_imgs_list=[]
    # em_imgs_list_ctf=[]
    # children_path = imgs_path_list[label_path_list == 2]
    # children_path_ctf = imgs_path_ctf_list[label_path_list == 2]
    # children=read_imgs_from_path_list(children_path)
    # # children_ctf=read_imgs_from_path_list(children_path)
    # children_ctf=read_imgs_from_path_list(children_path_ctf)
    # for i in range(children.shape[0]):
    #     em_imgs_list.append(EMNumPy.numpy2em(children[i]))
    #     em_imgs_list_ctf.append(EMNumPy.numpy2em(children_ctf[i]))
    # # children_em=EMNumPy.numpy2em(children)
    # # average=class_average_withali(children_em,iteration_times=3)
    # average=class_average(em_imgs_list)
    # average_np=to_int8(EMNumPy.em2numpy(average[0]))
    # Image.fromarray(average_np).save(save_path+'average.tif')
    # average_2=class_average_withali(images=em_imgs_list_ctf,ptcl_info=average[1],ref=average[0],xform= Transform(),focused=None)
    # average_np_ctf=to_int8(EMNumPy.em2numpy(average_2))
    # Image.fromarray(average_np_ctf).save(save_path+'average2.tif')


def save_augmented_imgs(
        augmented_imgs: torch.Tensor,
        weights: torch.Tensor,
        save_path: str,
        weight_bar: float = 0.0,
        img_name_prefix: str = 'augmented_img',
        images_per_row: int = 10,
        font_color: tuple = (255, 0, 0),  # 醒目的红色
        sorted_by_weight: bool = True,
):
    from PIL import Image, ImageDraw, ImageFont
    """
    筛选权重达标的图像，将权重值标注在图像上，然后将它们拼接成一张大图并保存。

    Args:
        augmented_imgs (torch.Tensor): 尺寸为 (N, 1, dim, dim) 的图像张量。
        weights (torch.Tensor): 尺寸为 (N,) 的权重张量，数值范围 0~1。
        save_path (str): 图像保存的目录路径。
        weight_bar (float, optional): 权重的筛选阈值。默认为 0.1。
        img_name_prefix (str, optional): 保存图像文件的前缀名。默认为 'augmented_img'。
        images_per_row (int, optional): 网格布局中每行的图像数量。默认为 10。
        font_color (tuple, optional): 标注文字的RGB颜色。默认为红色 (255, 0, 0)。
    """
    # 检查输入尺寸是否匹配
    if augmented_imgs.shape[0] != weights.shape[0]:
        raise ValueError(f"图像数量 ({augmented_imgs.shape[0]}) 和权重数量 ({weights.shape[0]}) 不匹配。")

    # 1. 按照 weight_bar 筛选图像和权重
    # 创建布尔掩码
    mask = weights > weight_bar
    filtered_imgs = augmented_imgs[mask]
    filtered_weights = weights[mask]

    num_filtered_imgs = filtered_imgs.shape[0]

    if num_filtered_imgs == 0:
        print(f"没有图像的权重超过 {weight_bar}，不保存任何图像。")
        return

    print(f"共有 {num_filtered_imgs} 张图像的权重超过 {weight_bar}，将进行处理。")

    # 2. 将权重标注在图像上
    annotated_imgs = []
    # 获取图像尺寸
    _, _, dim, _ = filtered_imgs.shape
    imgs_np=mrcs_to_int8(filtered_imgs.squeeze().cpu().numpy())  # 转换为 uint8 格式的 numpy 数组

    if sorted_by_weight:
        # 按照权重从大到小排序
        sorted_indices = torch.argsort(filtered_weights, descending=True)
    else:
        # 按照原始顺序
        sorted_indices = torch.arange(num_filtered_imgs)


    for i in sorted_indices:
        # img_tensor = filtered_imgs[i]  # Shape: (1, dim, dim)
        weight_value = filtered_weights[i].item()

        # 将单通道灰度图转换为 PIL Image 对象 (需要先转为 numpy)
        # 乘以 255 并转换为 uint8 类型
        # img_np = img_tensor.squeeze().cpu().numpy() * 255
        # pil_img = Image.fromarray(img_np.astype(np.uint8), 'L')
        pil_img =Image.fromarray(imgs_np[i].astype(np.uint8), 'L')
        # 转换为 RGB 模式以便添加彩色文字
        pil_img_rgb = pil_img.convert('RGB')

        # 在图像上绘制文字
        draw = ImageDraw.Draw(pil_img_rgb)
        text = f"{weight_value:.3f}"  # 格式化权重值为3位小数

        # 定义文字位置（左上角，留出一些边距）
        text_position = (5, 5)

        # 绘制文字
        draw.text(text_position, text, fill=font_color)

        # 将 PIL Image 对象转换回 Tensor
        # PIL (H, W, C) -> Numpy -> Tensor (C, H, W)
        back_to_np = np.array(pil_img_rgb)
        back_to_tensor = torch.from_numpy(back_to_np).permute(2, 0, 1)  # / 255.0 if normalization is needed
        annotated_imgs.append(back_to_tensor)

    if not annotated_imgs:
        print("没有可供处理的图像。")
        return

    # 将标注好的图像列表堆叠成一个 batch
    annotated_batch = torch.stack(annotated_imgs)

    # 3. 检查并创建保存目录
    os.makedirs(save_path, exist_ok=True)

    # 4. 组合图像并保存
    save_file_path = os.path.join(save_path, f"{img_name_prefix}.png")

    # 使用 torchvision.utils.save_image 直接生成网格图并保存
    save_image(
        annotated_batch.float() / 255.0,  # save_image 需要 [0,1] 范围的浮点数张量
        save_file_path,
        nrow=images_per_row,
        normalize=False  # 因为我们已经处理好了范围，不需要它再标准化
    )
    # print(f"图像已成功保存至: {save_file_path}")


def get_imgs_by_ids(dataloader, target_ids, pose_items_id,min_id=0,name='aug1'):
    dataset = dataloader.dataset
    results = []

    for idx in target_ids:
        results.append(dataset[pose_items_id[idx]+min_id][name].cpu().numpy())

    results =torch.from_numpy( np.asarray(results))  # 转换为形状 (N, 1, dim, dim)
    return results

if __name__ == '__main__':
    get_averages_with_labels()
