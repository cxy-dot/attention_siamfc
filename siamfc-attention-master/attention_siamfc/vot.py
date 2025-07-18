from __future__ import absolute_import, division, print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
from PIL import Image
from vot.dataset import Dataset
import time
from vot.region import Rectangle, Region, RegionType

from vot import analysis

from vot.utilities import to_number


def show_frame(image, boxes, legends=None, colors=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    # 默认颜色列表
    default_colors = ['white', 'red', 'lime', 'blue', 'cyan',
                      'magenta', 'yellow', 'orange', 'purple',
                      'brown', 'pink']

    colors = colors or default_colors

    for i, box in enumerate(boxes):
        if box is None:
            continue

        # 处理不同格式的box
        if hasattr(box, 'bounds'):  # VOT Rectangle对象
            x, y, w, h = box.bounds
        elif isinstance(box, (list, np.ndarray)):  # [x,y,w,h]格式
            x, y, w, h = box
        else:
            raise ValueError(f"Unsupported box format: {type(box)}")

        plt.gca().add_patch(Rectangle(
            (x, y), w, h,
            linewidth=3,
            edgecolor=colors[i % len(colors)],
            facecolor='none',
            label=legends[i] if legends and i < len(legends) else None
        ))

    if legends:
        plt.legend(loc='upper right', fontsize=12)

    plt.axis('off')
    plt.tight_layout()
    plt.show()

class ExperimentVOT(object):
    r"""Experiment pipeline and evaluation toolkit for VOT dataset.

    Args:
        root_dir (string): Root directory of VOT dataset.
        version (string): Specify the benchmark version (e.g., '2018').
        result_dir (string, optional): Directory for storing tracking results.
        report_dir (string, optional): Directory for storing performance reports.
    """

    def __init__(self, root_dir, version='2018',
                 result_dir='results', report_dir='reports'):
        super(ExperimentVOT, self).__init__()

        # Initialize VOT workspace
        self.workspace = Dataset.load(root_dir, "vot2018")
        self.version = version
        self.result_dir = os.path.join(result_dir, 'VOT' + version)
        self.report_dir = os.path.join(report_dir, 'VOT' + version)

        # VOT-specific parameters
        self.nbins_iou = 21
        self.nbins_ar = 21  # Accuracy-Robustness plots

    def run(self, tracker, visualize=False):
        print('Running tracker %s on VOT%s...' % (tracker.name, self.version))

        for sequence in self.workspace:
            seq_name = sequence.name
            print('--Sequence: %s' % seq_name)

            # Skip if results exist
            record_file = os.path.join(
                self.result_dir, tracker.name, '%s.txt' % seq_name)
            if os.path.exists(record_file):
                print('  Found results, skipping', seq_name)
                continue

            # Initialize tracker on first frame
            frame = sequence.frame(0)
            image = np.array(Image.open(frame.filename()))
            tracker.initialize(image, sequence.groundtruth(0))

            # Tracking loop
            boxes = []
            times = []

            for frame in sequence.frames():
                if frame.index == 0:
                    boxes.append(sequence.groundtruth(0))
                    times.append(0)
                    continue

                image = np.array(Image.open(frame.filename()))
                start_time = time.time()
                box = tracker.track(image)
                times.append(time.time() - start_time)
                boxes.append(box)

            # Record results
            self._record(record_file, boxes, times)

    def report(self, tracker_names):
        assert isinstance(tracker_names, (list, tuple))

        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
        report_file = os.path.join(report_dir, 'performance.json')

        performance = {}

        for name in tracker_names:
            print('Evaluating', name)
            performance.update({name: {
                'overall': {},
                'seq_wise': {}}})

            # VOT-specific metrics
            accuracies = []
            robustness = []
            speeds = []

            for sequence in self.workspace:
                seq_name = sequence.name
                record_file = os.path.join(
                    self.result_dir, name, '%s.txt' % seq_name)

                if not os.path.exists(record_file):
                    continue

                # Load results
                boxes = np.loadtxt(record_file, delimiter=',')
                gt = [sequence.groundtruth(i) for i in range(len(sequence))]

                # Calculate accuracy (average overlap)
                overlaps = self._calc_overlaps(boxes, gt)
                accuracy = np.mean(overlaps)
                accuracies.append(accuracy)

                # Calculate robustness (failure rate)
                failures = sum(o < 0.1 for o in overlaps)  # threshold can be adjusted
                robustness.append(failures / len(sequence))

                # Calculate speed
                time_file = os.path.join(
                    self.result_dir, name, 'times/%s_time.txt' % seq_name)
                if os.path.isfile(time_file):
                    times = np.loadtxt(time_file)
                    speeds.append(np.mean(1. / times[times > 0]))

                # Store sequence-wise performance
                performance[name]['seq_wise'].update({seq_name: {
                    'accuracy': accuracy,
                    'robustness': failures / len(sequence),
                    'speed_fps': speeds[-1] if len(speeds) > 0 else -1}})

            # Calculate overall performance
            if accuracies:
                ar_score = np.mean(accuracies) * (1 - np.mean(robustness))
                performance[name]['overall'].update({
                    'accuracy': np.mean(accuracies),
                    'robustness': np.mean(robustness),
                    'ar_score': ar_score,
                    'speed_fps': np.mean(speeds) if speeds else -1})

        # Save report
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)

        # Plot AR curves
        self.plot_curves(tracker_names)

        return performance


    def show(self, tracker_names, seq_names=None, play_speed=1):
        if seq_names is None:
            sequences = self.workspace
        else:
            sequences = [s for s in self.workspace if s.name in seq_names]

        play_speed = int(round(play_speed))
        assert play_speed > 0

        for sequence in sequences:
            print('Showing results on %s...' % sequence.name)

            # Load all tracking results
            records = {}
            for name in tracker_names:
                record_file = os.path.join(
                    self.result_dir, name, '%s.txt' % sequence.name)
                if os.path.exists(record_file):
                    records[name] = np.loadtxt(record_file, delimiter=',')

            # Display results
            for frame in sequence.frames():
                if not frame.index % play_speed == 0:
                    continue

                image = np.array(Image.open(frame.filename()))
                boxes = [sequence.groundtruth(frame.index)] + [
                    records[name][frame.index] for name in tracker_names
                    if name in records]

                show_frame(image, boxes,
                           legends=['GroundTruth'] + tracker_names,
                           colors=['w', 'r', 'g', 'b', 'c', 'm', 'y',
                                   'orange', 'purple', 'brown', 'pink'])

    def _record(self, record_file, boxes, times):
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')

        time_dir = os.path.join(record_dir, 'times')
        if not os.path.isdir(time_dir):
            os.makedirs(time_dir)
        time_file = os.path.join(time_dir, os.path.basename(
            record_file).replace('.txt', '_time.txt'))
        np.savetxt(time_file, times, fmt='%.8f')

    def _calc_overlaps(self, boxes, gt):
        overlaps = []
        for b, g in zip(boxes, gt):
            if isinstance(b, (list, tuple, np.ndarray)):
                b = Rectangle(*b)
            if isinstance(g, (list, tuple, np.ndarray)):
                g = Rectangle(*g)
            overlaps.append(b.overlap(g))
        return overlaps

    def plot_curves(self, tracker_names):
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        report_file = os.path.join(report_dir, 'performance.json')

        with open(report_file) as f:
            performance = json.load(f)

        ar_file = os.path.join(report_dir, 'ar_plot.png')

        # Prepare data
        acc = []
        rob = []
        names = []
        for name in tracker_names:
            if name in performance and 'overall' in performance[name]:
                acc.append(performance[name]['overall']['accuracy'])
                rob.append(performance[name]['overall']['robustness'])
                names.append(name)

        # Plot AR graph
        fig, ax = plt.subplots()
        for i, name in enumerate(names):
            ax.scatter(rob[i], acc[i], label=name)

        ax.set_xlabel('Robustness (failure rate)')
        ax.set_ylabel('Accuracy (average overlap)')
        ax.set_title('Accuracy-Robustness plot for VOT%s' % self.version)
        ax.legend()
        ax.grid(True)

        print('Saving AR plot to', ar_file)
        fig.savefig(ar_file, dpi=300)