# @Author : Written/modified by Yang Yan
# @Time   : 2022/6/20 下午7:08
# @E-mail : yanyang98@yeah.net
# @Function :
import os
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score


# def calculate_interval_proportions(input_dict, num_intervals=10):
#     # 获取所有的值
#     values = list(input_dict.values())
#
#     # 计算最小值和最大值
#     min_value = min(values)
#     max_value = max(values)
#
#     # 计算区间宽度
#     interval_width = (max_value - min_value) / num_intervals
#
#     # 初始化一个字典来存储每个区间的计数
#     interval_counts = {i: 0 for i in range(num_intervals)}
#
#     # 遍历所有值，计算它们所在的区间
#     for value in values:
#         if value == max_value:
#             interval_counts[num_intervals - 1] += 1
#         else:
#             interval_index = int((value - min_value) // interval_width)
#             interval_counts[interval_index] += 1
#
#     # 计算每个区间的比例
#     total_values = len(values)
#     interval_proportions = {i: count / total_values for i, count in interval_counts.items()}
#
#     # 创建一个新字典，key与输入一致，value为对应值所在区间的比例
#     result_dict = {}
#     for key, value in input_dict.items():
#         if value == max_value:
#             interval_index = num_intervals - 1
#         else:
#             interval_index = int((value - min_value) // interval_width)
#         result_dict[key] = interval_proportions[interval_index]
#
#     return result_dict

def calculate_interval_proportions(input_dict, num_intervals=10):
    # 获取所有的值
    values = list(input_dict.values())
    values_sum=sum(values)
    values_num=len(values)


    # 创建一个新字典，key与输入一致，value为对应值所在区间的比例
    result_dict = {}
    for key, value in input_dict.items():
        result_dict[key] = values_num*value/values_sum

    return result_dict

def get_mean_error_distribution(mean_error_dict):
    mean_error_good_dict={name:mean_error for name,mean_error in mean_error_dict.items() if not name.lower().endswith('_bad')}
    mean_error_bad_dict={name:mean_error for name,mean_error in mean_error_dict.items() if name.lower().endswith('_bad')}
    dis_good=calculate_interval_proportions(mean_error_good_dict)
    dis_bad=calculate_interval_proportions(mean_error_bad_dict)
    # mean_error_dict_all={'good':dis_good,'bad':dis_bad}
    mean_error_dict_all={**dis_good,**dis_bad}
    return mean_error_dict_all

def balanced_mask(target_y_t_score, bar_score,random_seed=0):
    np.random.seed(random_seed)
    # 计算满足两个条件的掩码
    high_mask = target_y_t_score > bar_score[1]
    low_mask = target_y_t_score <= bar_score[0]

    # 计算满足每个条件的元素数量
    high_count = np.sum(high_mask)
    low_count = np.sum(low_mask)

    # 确定需要保留的元素数量
    keep_count = min(high_count, low_count)
    if keep_count==0:
        return np.ones_like(target_y_t_score, dtype=bool)

    # 如果高分数的元素过多，随机选择部分元素
    if high_count > keep_count:
        high_indices = np.where(high_mask)[0]
        keep_high_indices = np.random.choice(high_indices, keep_count, replace=False)
        high_mask = np.zeros_like(high_mask)
        high_mask[keep_high_indices] = True

    # 如果低分数的元素过多，随机选择部分元素
    if low_count > keep_count:
        low_indices = np.where(low_mask)[0]
        keep_low_indices = np.random.choice(low_indices, keep_count, replace=False)
        low_mask = np.zeros_like(low_mask)
        low_mask[keep_low_indices] = True

    # 合并两个条件的结果
    mask = high_mask | low_mask

    return mask
def evaluate_fit(score_predicted_data, score_truth_data):
    """
    评估预测数据与真实数据对 y=x 的拟合程度。

    参数:
    confidence_predicted_data : np.ndarray
        预测的置信度数据
    confidence_truth_data : np.ndarray
        真实的置信度数据

    返回:
    dict
        包含多个评估指标的字典
    """
    # 确保输入数据的形状相同
    assert score_predicted_data.shape == score_truth_data.shape, "输入数组的形状必须相同"

    # 计算均方误差 (MSE)
    mse = mean_squared_error(score_truth_data, score_predicted_data)

    # 计算均方根误差 (RMSE)
    rmse = np.sqrt(mse)

    # 计算决定系数 (R-squared)
    r2 = r2_score(score_truth_data, score_predicted_data)

    # 计算皮尔逊相关系数
    pearson_corr, _ = stats.pearsonr(score_truth_data.flatten(), score_predicted_data.flatten())

    # 计算斯皮尔曼等级相关系数
    spearman_corr, _ = stats.spearmanr(score_truth_data, score_predicted_data)

    # 计算平均绝对误差 (MAE)
    mae = np.mean(np.abs(score_predicted_data - score_truth_data))

    # 计算最大绝对误差
    max_error = np.max(np.abs(score_predicted_data - score_truth_data))

    return {
        "MSE": mse,
        "RMSE": rmse,
        "R-squared": r2,
        "Pearson Correlation": pearson_corr,
        "Spearman Correlation": spearman_corr,
        "MAE": mae,
        "Max Error": max_error
    }
def plot_mean_score_histogram(score_predicted_data, score_truth_data, result_dir, epoch='final'):
    # 确保输入数据的形状正确
    assert score_predicted_data.shape == score_truth_data.shape
    assert len(score_predicted_data.shape) == 1 or score_predicted_data.shape[1] == 1

    # 如果是2D数组，转换为1D
    if len(score_predicted_data.shape) == 2:
        score_predicted_data = score_predicted_data.flatten()
        score_truth_data = score_truth_data.flatten()

    # 创建10个区间
    bins = np.linspace(0, 1, 11)

    # 计算每个区间的平均真实置信度和标准差
    averages = []
    std_devs = []
    for i in range(10):
        mask = (score_predicted_data >= bins[i]) & (score_predicted_data < bins[i + 1])
        if np.sum(mask) > 0:
            avg = np.mean(score_truth_data[mask])
            std = np.std(score_truth_data[mask])  # 计算标准差
        else:
            avg = 0
            std = 0
        averages.append(avg)
        std_devs.append(std)

    # 创建图表对象
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制柱状图和误差线
    # bars = ax.bar(bins[:-1], averages, width=0.08, align='edge', yerr=std_devs, capsize=5, ecolor='red', alpha=0.7)
    bars = ax.bar(bins[:-1], averages, width=0.08, align='edge',  capsize=5, ecolor='red', alpha=0.7)

    # 设置标签和标题
    ax.set_xlabel('Predicted Score')
    ax.set_ylabel('Average True Score')
    # ax.set_title('Average True Confidence vs Predicted Confidence (Error bars: Standard Deviation)')
    ax.set_title('Average True Score vs Predicted Score')

    # 设置坐标轴范围和刻度
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))

    # 添加网格线
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 在每个柱子上方显示具体数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2f}',
                ha='center', va='bottom')

    plt.savefig(result_dir + '/figures/mean_score_interval_' + str(epoch) + '.svg')
    plt.close()
    return fig
def plot_mean_score_violin(scores_data_predicted_np, scores_data_true_np, result_dir, epoch='final'):
    fig, ax = plt.subplots(figsize=(8, 6))

    # 设置字体大小
    plt.rcParams.update({'font.size': 14})

    # 设置x轴刻度
    x_ticks = np.arange(0, 1.1, 0.1)
    # 存储每个区间的平均值
    mean_values = []
    # 存储小提琴图的位置
    violin_positions = []

    # 遍历每个x轴刻度范围
    for i, x_tick in enumerate(x_ticks[:-1]):
        next_x_tick = x_ticks[i + 1]
        mid_point = (x_tick + next_x_tick) / 2
        violin_positions.append(mid_point)

        if i == len(x_ticks) - 2:
            mask = (scores_data_true_np >= x_tick) & (scores_data_true_np <= next_x_tick)
        else:
            mask = (scores_data_true_np >= x_tick) & (scores_data_true_np < next_x_tick)

        # 获取当前范围内的预测值
        predicted_values = scores_data_predicted_np[mask]

        # 计算当前范围内的平均值
        mean_value = np.mean(predicted_values)
        mean_values.append(mean_value)

        # 绘制小提琴图
        if len(predicted_values) > 0:
            violin_parts = ax.violinplot([predicted_values], positions=[mid_point], showmeans=False, showmedians=False,
                                         showextrema=False, widths=0.04)



            # 设置小提琴图颜色和透明度
            for vp in violin_parts['bodies']:
                vp.set_facecolor('lightblue')
                vp.set_edgecolor('none')
                vp.set_alpha(0.7)

    # 绘制折线图
    ax.plot(violin_positions, mean_values, color='red', marker='o')

    # 在节点上显示平均值数字
    for x, y in zip(violin_positions, mean_values):
        ax.annotate(f'{y:.2f}', (x, y), xytext=(5, 5), textcoords='offset points')

    # 设置x轴刻度标签
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{x_tick:.1f}' for x_tick in x_ticks])

    # 设置y轴范围
    ax.set_ylim(0, 1.1)

    # 设置坐标轴标签和标题
    ax.set_xlabel('True Score')
    ax.set_ylabel('Predicted Score')
    ax.set_title('Violin Plot')

    plt.savefig(result_dir + '/figures/mean_score_interval_violin_' + str(epoch) + '.svg')
    plt.close()
    return fig
def plot_score_distribution(score_data, result_dir, epoch='final', score_bar=[0.4, 0.7]):
    # 确保输入数据是一维数组
    score_data = score_data.flatten()

    # 创建直方图数据
    hist, bin_edges = np.histogram(score_data, bins=np.arange(0, 1.1, 0.1))

    # 计算每个区间的数量
    low = np.sum(score_data < score_bar[0])
    mid = np.sum((score_data >= score_bar[1]) & (score_data < score_bar[0]))
    high = np.sum(score_data >= score_bar[1])

    # 创建自定义颜色映射
    colors = ['#3498db', '#f39c12', '#2ecc71']  # 蓝色、橙色、绿色
    n_bins = 3
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

    # 为每个柱子分配颜色
    bar_colors = [cmap(0)] * 10 + [cmap(1)] * 1 + [cmap(2)] * 1
    for i in range(10):
        if bin_edges[i] >= score_bar[1] and bin_edges[i] < score_bar[0]:
            bar_colors[i] = cmap(1)
        if bin_edges[i] >= score_bar[1]:
            bar_colors[i] = cmap(2)

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制直方图
    bars = ax.bar(bin_edges[:-1], hist, width=0.08, align='edge', color=bar_colors, edgecolor='white')

    # 添加标签和标题
    ax.set_xlabel('Score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Score Distribution (Bar: {score_bar})', fontsize=14)

    # 在每个柱子上方显示数量
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=8)

    # 添加图例
    if score_bar[0] != score_bar[1]:
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=cmap(0), edgecolor='none', label=f'Low: {low}'),
            plt.Rectangle((0, 0), 1, 1, facecolor=cmap(1), edgecolor='none', label=f'Mid: {mid}'),
            plt.Rectangle((0, 0), 1, 1, facecolor=cmap(2), edgecolor='none', label=f'High: {high}')
        ]
    else:
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=cmap(0), edgecolor='none', label=f'Low: {low}'),
            plt.Rectangle((0, 0), 1, 1, facecolor=cmap(2), edgecolor='none', label=f'High: {high}')
        ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.01, 1), fontsize=10)

    # 调整x轴刻度
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xlim(-0.23, 1.05)  # 稍微扩大x轴范围，以便柱子不会紧贴边缘

    # 显示网格线
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 调整图形布局
    plt.tight_layout()

    # 调整左边距和上边距
    plt.subplots_adjust(left=0.08, top=0.9)
    plt.savefig(result_dir + '/figures/num_interval_' + str(epoch) + '.svg')
    plt.close()
    return fig
def plot_score_interval_precision_recall(score_predicted_data, score_truth_data, result_dir, epoch='final', score_bar=[0.4, 0.7]):
    """
    绘制confidence_predicted_data在各个confidence间隔内二分类预测的准确度柱状图

    参数：
    - confidence_predicted_data (ndarray): n*1大小的预测置信度数据
    - confidence_truth_data (ndarray): n*1大小的真实标签数据
    - result_dir (str): 结果保存目录
    - epoch (str): 当前epoch，默认为'final'
    - confidence_bar (float): 0.5到1之间的数字，用于区分置信度的阈值
    """
    # 将confidence_predicted_data和confidence_truth_data转换为一维数组
    score_predicted_data = score_predicted_data.flatten()
    score_truth_data = score_truth_data.flatten()

    # 计算每个置信度间隔内的预测准确度和召回率
    bins = np.arange(0, 1.1, 0.1)
    precision = []
    recall = []
    counts = []
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
        if i == len(bins) - 2:
            upper += 0.01
        mask = (score_predicted_data >= lower) & (score_predicted_data < upper)

        if mask.sum() > 0:
            predicted_labels = (score_predicted_data[mask] >= score_bar[1]).astype(int)
            truth_labels = (score_truth_data[mask] >= score_bar[1]).astype(int)
            tp = np.sum(predicted_labels == truth_labels)


            precision.append(tp / predicted_labels.shape[0] if predicted_labels.shape[0] > 0 else 0)

            counts.append(mask.sum())
        else:
            precision.append(0)
            counts.append(0)
        mask_truth = (score_truth_data >= lower) & (score_truth_data < upper)
        if mask_truth.sum() > 0:
            predicted_labels2 = (score_predicted_data[mask_truth] >= score_bar[1]).astype(int)
            truth_labels2 = (score_truth_data[mask_truth] >= score_bar[1]).astype(int)
            tp2 = np.sum(predicted_labels2 == truth_labels2)
            truth_labels_interval=(score_truth_data[mask_truth] >= score_bar[1]).astype(int)
            recall.append(tp2 / truth_labels_interval.shape[0] if truth_labels_interval.shape[0] > 0 else 0)
        else:
            recall.append(0)

    # 创建自定义颜色映射
    colors = ['#3498db', '#f39c12', '#2ecc71']  # 蓝色、橙色、绿色
    n_bins = 2
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

    # 创建一个fig对象
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制柱状图
    bar_width = 0.04
    x = (bins[:-1] + bins[1:]) / 2  # 计算每个区间的中点
    rects1 = ax.bar(x - bar_width/2, precision, width=bar_width, alpha=0.7, color=cmap(0), label='Precision')
    rects2 = ax.bar(x + bar_width/2, recall, width=bar_width, alpha=0.7, color=cmap(1), label='Recall')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)

    # 为大于等于confidence_bar的柱子添加黑色边框
    for i in range(len(bins) - 1):
        if bins[i] >= score_bar[1]:
            ax.bar(x[i] - bar_width / 2, precision[i], width=bar_width, alpha=0.7, color='none', edgecolor='black',
                   linewidth=2)
            ax.bar(x[i] + bar_width / 2, recall[i], width=bar_width, alpha=0.7, color='none', edgecolor='black',
                   linewidth=2)

    # 添加标签和标题
    ax.set_xlabel('Score')
    ax.set_ylabel('Precision / Recall')
    ax.set_title('Precision and Recall vs Score')
    ax.set_ylim(0, 1.1)  # 设置y轴范围避免重合
    ax.set_xticks(bins)  # 设置x轴刻度以0.1为间隔
    ax.set_xticklabels([f'{b:.1f}' for b in bins])  # 设置x轴刻度标签

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)

    # 添加图例
    ax.legend()

    # 显示图形
    plt.tight_layout()
    plt.savefig(result_dir + f'/figures/score_interval_precision_recall_{epoch}.svg', dpi=300, bbox_inches='tight')
    plt.close(fig)  # 关闭图形，释放内存
    return fig
def plot_score_interval_inf(score_data, labels_p, labels_t, result_dir, epoch='final', bin_num=10):
    labels_data = []
    accuracy_data = []
    precision_data = []
    interval_num_data = []
    interval_pos_num_data = []

    if labels_t is None:
        calculate_acc = False
    elif -1 in labels_t:
        calculate_acc = False
    else:
        calculate_acc = True
    for i in range(bin_num):
        score_min = i / bin_num
        score_max = (i + 1) / bin_num
        my_index = (score_data >= score_min) & (score_data < score_max)
        # confidence_data_i = confidence_data[my_index]
        labels_p_i = labels_p[my_index]

        interval_num_data.append(len(labels_p_i))
        pos_num = sum(labels_p_i == 1)
        interval_pos_num_data.append(pos_num)
        labels_data.append(str(score_min) + '-' + str(score_max))

        if calculate_acc and len(labels_p_i) > 0:
            labels_t_i = labels_t[my_index]
            accuracy_data_i = np.mean(labels_p_i == labels_t_i)
            accuracy_data.append(accuracy_data_i)
            precision_correct_num = sum(labels_p_i[labels_p_i == 1] == labels_t_i[labels_p_i == 1])
            if pos_num>0:
                precision_data.append((precision_correct_num / pos_num))
            else:
                precision_data.append(0)
        else:
            accuracy_data.append(0)
            precision_data.append(0)
    if not os.path.exists(result_dir + '/figures/'):
        os.makedirs(result_dir + '/figures/')
    my_width = 0.4
    x = np.arange(bin_num)
    fig_num, ax_num = plt.subplots()
    fig_num.set_size_inches((14, 8))

    num_bar = ax_num.bar(x - my_width / 2, interval_num_data, my_width,
                         label='particles number (all ' + str(len(labels_p)) + ')')
    ax_num.bar_label(num_bar, label_type='edge')

    num_bar_pos = ax_num.bar(x + my_width / 2, interval_pos_num_data, my_width,
                             label='positive particles number (all ' + str(sum(interval_pos_num_data)) + ')')
    ax_num.bar_label(num_bar_pos, label_type='edge')
    ax_num.set_xticks(x)
    ax_num.set_xticklabels(labels_data)
    ax_num.set_xlabel('score')
    ax_num.set_title('Particles num statistics')
    ax_num.legend()
    fig_num.savefig(result_dir + '/figures/score_interval_num_' + str(epoch) + '.svg')

    if calculate_acc:
        fig_acc, ax_acc = plt.subplots()
        fig_acc.set_size_inches((14, 8))
        acc_bar = ax_acc.bar(x - my_width / 2, accuracy_data, my_width, label='classification accuracy')
        ax_acc.bar_label(acc_bar, labels=['%.3f' % acc if acc != 0 else str(0) for acc in accuracy_data],
                         label_type='edge')
        p_bar = ax_acc.bar(x + my_width / 2, precision_data, my_width, label='classification precision')
        ax_acc.bar_label(p_bar, labels=['%.3f' % p if p != 0 else str(0) for p in precision_data], label_type='edge')
        ax_acc.set_xticks(x)
        ax_acc.set_xticklabels(labels_data)
        ax_acc.set_xlabel('score')
        ax_acc.set_title('Classification performance')
        ax_acc.legend()
        fig_acc.savefig(result_dir + '/figures/score_interval_acc_' + str(epoch) + '.svg')
    else:
        fig_acc = None
    plt.close(fig_num)
    if fig_acc is not None:
        plt.close(fig_acc)

    return fig_num, fig_acc


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_clustering_labels(labels_path, epoch, clustering_labels):
    if not os.path.exists(labels_path):
        os.makedirs(labels_path + '/')
    np.save(labels_path + '/predict_labels_epoch' + str(epoch) + '.npy', clustering_labels)


def save_acc_data(acc, acc_best, nmi, nmi_best, tb_writer, acc_sum, nmi_sum, clustering_times, epoch, out_path):
    if acc > acc_best:
        acc_best = acc
    if nmi > nmi_best:
        nmi_best = nmi
    acc_sum = acc_sum + acc
    nmi_sum = nmi + nmi_sum
    clustering_times = clustering_times + 1
    tb_writer.add_scalar("accuracy:", acc, epoch)
    tb_writer.add_scalar("nmi:", nmi, epoch)
    tb_writer.add_scalar("mean accuracy:", acc_sum / clustering_times,
                         clustering_times)
    tb_writer.add_scalar("mean NMI:", nmi_sum / clustering_times,
                         clustering_times)
    if not os.path.exists(out_path + 'acc_data/'):
        os.makedirs(out_path + 'acc_data/')
    with open(out_path + 'acc_data/' + 'acc_data.txt', 'w') as average_acc:
        average_acc.write('best accuracy' + str(acc_best))
        average_acc.write('\nbest NMI' + str(nmi_best))
        average_acc.write('\naverage accuracy' + str(acc_sum / clustering_times))
        average_acc.write('\naverage NMI' + str(nmi_sum / clustering_times))
    return acc_best, nmi_best, acc_sum, nmi_sum, clustering_times


def save_trained_model(model, optimizer, epoch, save_path):
    import torch
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                'epoch': epoch + 1}, save_path + 'epoch' + str(epoch) + '_model.pth.tar')


def save_averages(average_imgs_all, average_generated_imgs=None, run_root_dir=None, epoch=0):
    import torch
    import mrcfile
    if not os.path.exists(run_root_dir + 'averages/'):
        os.makedirs(run_root_dir + 'averages/')
    save_image(torch.unsqueeze(torch.from_numpy(average_imgs_all), 1),
               run_root_dir + 'averages/clustering_result_' + str(epoch) + '.png')
    if average_generated_imgs is not None:
        save_image(torch.unsqueeze(torch.from_numpy(average_generated_imgs), 1),
                   run_root_dir + 'averages/generated_clustering_result_' + str(epoch) + '.png')
        # projectons_file = mrcfile.new(
        #     run_root_dir + 'averages/generated_clustering_averages_' + str(epoch) + '.mrcs',
        #     average_generated_imgs, overwrite=True)
    projectons_file = mrcfile.new(
        run_root_dir + 'averages/clustering_averages_' + str(epoch) + '.mrcs',
        average_imgs_all, overwrite=True)
    projectons_file.close()


def confusion_matrix(predictions, gt, class_names, output_file=None):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)

    fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix, cmap='Blues')
    axes.set_xticks([i for i in range(len(class_names))])
    axes.set_yticks([i for i in range(len(class_names))])
    axes.set_xticklabels(class_names, ha='right', fontsize=8, rotation=40)
    axes.set_yticklabels(class_names, ha='right', fontsize=8)

    for (i, j), z in np.ndenumerate(confusion_matrix):
        if i == j:
            axes.text(j, i, '%d' % (100 * z), ha='center', va='center', color='white', fontsize=6)
        else:
            pass

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def free_mem():
    import os
    result = os.popen("fuser -v /dev/nvidia*").read()
    results = result.split()
    for pid in results:
        os.system(f"kill -9 {int(pid)}")


def numsCheng(i):
    for m in range(100000):
        m = i * 2
        pass
    return i * 2, 2


def multi_process_test():
    import torch
    import time
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs
    from multiprocessing.pool import Pool

    print("start")
    time1 = time.time()
    nums_list = range(100000)
    pool = Pool(processes=10)
    result = pool.map(numsCheng, nums_list)
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程的退出
    print("end")
    # print(result)
    time2 = time.time()
    print("计算用时：", time2 - time1)

    # for i,j in result:
    #     print(i,j)


if __name__ == '__main__':
    a=[0.01,0.12,0.1,0.13,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    b=[0.0001,0.01,0.005,0.12,0.1,0.13,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    bar=0.6
    mask=balanced_mask(np.array(a),bar)
    mask2=balanced_mask(np.array(b),bar)
    mask3=balanced_mask(np.array(a),bar,random_seed=2)
    pass
    # multi_process_test()
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import seaborn as sns
    # import pandas as pd
    #
    # bin_num = 10
    # confidence_data = np.random.rand(100)
    # labels_p = np.random.randint(0, 10, 100)
    # labels_t = np.random.randint(0, 10, 100)
    # # labels_p = np.random.randint(0, 10, 100)
    # # labels_t = np.random.randint(0, 10, 100)
    # # confidence_data = confidence_data[:, labels_p == labels_t]
    # labels_p = labels_p[labels_p == labels_t]
    # # confidence_data = np.max(confidence_data, axis=1)
    # confidence_data = np.clip(confidence_data, 0, 1)
    # bins = np.linspace(0, 1, bin_num)
    # labels = np.digitize(confidence_data, bins)
    # data = pd.DataFrame({'confidence': confidence_data, 'label': labels_p, 'bin': labels})
    # data = data.groupby('bin').apply(lambda x: np.mean(x['label'] == x['label'].mode().values[0]))
    # data = data.reset_index()
    # data.columns = ['bin', 'accuracy']
    # plt.figure()
    # sns.barplot(x='bin', y='accuracy', data=data)
    # # plt.show()
    # plt.savefig('/yanyang2/projects/results/particle_classification/inference_test/2024_2_25_P954W7J29/test.png')

    # free_mem()
    # multi_process_test()
    # l=['a','b','c']
    # l.append(None)
    # print(l)
    # print('a','b')
# def save_classified_particles(run_root_dir, epoch, generated_imgs, clustering_labels, class_number_arry,cluster_num):
#     if not os.path.exists(run_root_dir + 'class_single_particles/epoch' + str(epoch) + '/'):
#         os.makedirs(run_root_dir + 'class_single_particles/epoch' + str(epoch) + '/')
#     if not os.path.exists(run_root_dir + 'generated_class_single_particles/epoch' + str(epoch) + '/'):
#         os.makedirs(run_root_dir + 'generated_class_single_particles/epoch' + str(epoch) + '/')
#     if Running_Paras.is_save_clustered_single_particles:
#         for i in range(cluster_num):
#             # class_single_particles = raw_mrcArrays[clustering_labels == i]
#             generated_class_single_particles = generated_imgs[clustering_labels == i]
#             generated_class_single_particles = np.squeeze(generated_class_single_particles)
#             class_particles = mrcfile.new(
#                 run_root_dir + 'generated_class_single_particles/epoch' + str(epoch) + '/' + 'class_' + str(
#                     i) + "_" + str(class_number_arry[i]) + '.mrcs',
#                 generated_class_single_particles, overwrite=True)
#             class_particles.close()
#         evaluate_utils.classify_mrcs(clustering_labels, cluster_num,
#                                      Running_Paras.path_for_data_classify,
#                                      run_root_dir + 'class_single_particles/epoch' + str(epoch) + '/')
