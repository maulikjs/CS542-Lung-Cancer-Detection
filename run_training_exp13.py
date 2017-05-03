import os
import tensorflow as tf
import csv
import numpy as np

################################################################################
# Training setup
################################################################################
# The data and params
DATA_DIR = './data/preprocessed_reshaped2/'
DATA_MEAN_FILE = './data/preprocessed_reshaped2_mean.npy'
LABLE_FILE_TRAIN = './data/stage1_labels.csv'
LABLE_FILE_TEST = './data/stage1_labels_stage1test.csv'

NUM_EPOCHES = 10000
EXP_NAME = 'exp13_crops'
LOG_DIR = './train/%s/tb/' % EXP_NAME
LOG_FILE = './train/%s/log.txt' % EXP_NAME

SAVE_EPOCHES = 5
SAVE_DIR = './train/%s/save/' % EXP_NAME

WEIGHT_DECAY = 1e-3

# The data shape
N = 64
DATA_Z, DATA_Y, DATA_X = 64, 64, 64
CROP_Z, CROP_Y, CROP_X = 54, 54, 54

# make the directories
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# Set output log file
log_f = open(LOG_FILE, 'w')
def print_and_log(*argv):
    print(*argv, file=log_f, flush=True)
    print(*argv, flush=True)

################################################################################
# Load data
################################################################################

mean_data = np.load(DATA_MEAN_FILE)

# A. Load training data
# Load image and label files
sample_list_trn = []
with open(LABLE_FILE_TRAIN) as f:
    for name, label in csv.reader(f):
        sample_list_trn.append((name, float(label)))
# number of samples
num_sample_trn = len(sample_list_trn)
print_and_log('number of training samples:', num_sample_trn)

# Shuffle the sample list
np.random.shuffle(sample_list_trn)

# Pre-load all batches into memory
trn_image_array = np.zeros((num_sample_trn, DATA_Z, DATA_Y, DATA_X, 1), np.float32)
trn_label_array = np.zeros((num_sample_trn, 1), np.float32)
for n in range(num_sample_trn):
    name, label = sample_list_trn[n]
    trn_label_array[n, 0] = label
    d = np.load(os.path.join(DATA_DIR, name+'.npz'))
    trn_image_array[n, ..., 0] = d['pix_resampled'] - mean_data
    d.close()

# B. Load test data
# Load image and label files
sample_list_tst = []
with open(LABLE_FILE_TEST) as f:
    for name, label in csv.reader(f):
        sample_list_tst.append((name, float(label)))
# number of samples
num_sample_tst = len(sample_list_tst)
print_and_log('number of test samples:', num_sample_tst)

# Pre-load all batches into memory
tst_image_array = np.zeros((num_sample_tst, DATA_Z, DATA_Y, DATA_X, 1), np.float32)
tst_label_array = np.zeros((num_sample_tst, 1), np.float32)
for n in range(num_sample_tst):
    name, label = sample_list_tst[n]
    tst_label_array[n, 0] = label
    d = np.load(os.path.join(DATA_DIR, name+'.npz'))
    tst_image_array[n, ..., 0] = d['pix_resampled'] - mean_data
    d.close()

# get random crops from the data
cropped_image_array_buffer = np.zeros((N, CROP_Z, CROP_Y, CROP_X, 1), np.float32)
def get_random_crops(batch_image_array):
    N_actual = len(batch_image_array)
    cropped_image_array = cropped_image_array_buffer[:N_actual]
    
    begin_z = np.random.randint(DATA_Z-CROP_Z, size=N_actual)
    begin_y = np.random.randint(DATA_Y-CROP_Y, size=N_actual)
    begin_x = np.random.randint(DATA_X-CROP_X, size=N_actual)
    end_z = begin_z + CROP_Z
    end_y = begin_y + CROP_Y
    end_x = begin_x + CROP_X
    for n in range(N_actual):
        cropped_image_array[n] = \
            batch_image_array[n,
                              begin_z[n]:end_z[n],
                              begin_y[n]:end_y[n],
                              begin_x[n]:end_x[n]]
    return cropped_image_array
    

################################################################################
# The network
################################################################################

# An initializer that keeps the variance the same across layers
def scaling_initializer():
    def _initializer(shape, dtype=None, partition_info=None):
        fan_in = 1.0
        for dim in shape[:-1]:
            fan_in *= dim

        trunc_stddev = np.sqrt(2 / fan_in)
        return tf.truncated_normal(shape, 0., trunc_stddev, dtype)

    return _initializer

kernel_initializer = scaling_initializer()
kernel_regularizer = tf.contrib.layers.l2_regularizer(WEIGHT_DECAY)

def model(input_data, kernel_regularizer, is_training, scope='CNN3D', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        output = tf.layers.conv3d(input_data, 48, (3, 3, 3), (1, 1, 1), "VALID", name='conv1',
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer, activation=tf.nn.relu)
        output = tf.layers.conv3d(output, 48, (3, 3, 3), (1, 1, 1), "VALID", name='conv2',
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer, activation=tf.nn.relu)
        output = tf.layers.max_pooling3d(output, (2, 2, 2), (2, 2, 2), "VALID", name='pool2')

        output = tf.layers.conv3d(output, 64, (3, 3, 3), (1, 1, 1), "VALID", name='conv3',
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer, activation=tf.nn.relu)
        output = tf.layers.conv3d(output, 64, (3, 3, 3), (1, 1, 1), "VALID", name='conv4',
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer, activation=tf.nn.relu)
        output = tf.layers.max_pooling3d(output, (2, 2, 2), (2, 2, 2), "VALID", name='pool4')

        output = tf.layers.conv3d(output, 96, (3, 3, 3), (1, 1, 1), "VALID", name='conv5',
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer, activation=tf.nn.relu)
        output = tf.layers.conv3d(output, 96, (3, 3, 3), (1, 1, 1), "VALID", name='conv6',
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer, activation=tf.nn.relu)
        output = tf.layers.max_pooling3d(output, (2, 2, 2), (2, 2, 2), "VALID", name='pool6')
        print_and_log('shape before fully connected layer:', output.get_shape())

        # Flatten the output
        output = tf.reshape(output, (-1, np.prod(output.get_shape().as_list()[1:])))

        output = tf.layers.dense(output, 256, name='dense7',
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer, activation=tf.nn.relu)
        output = tf.layers.dropout(output, training=is_training, name='drop7')
        output = tf.layers.dense(output, 1, name='dense8',
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer)

    return output

# A. Training network
data_trn = tf.placeholder(tf.float32, [None, CROP_Z, CROP_Y, CROP_X, 1], name='data')
output_trn = model(data_trn, kernel_regularizer=kernel_regularizer, is_training=True)

# Loss
ground_truth_trn = tf.placeholder(tf.float32, [None, 1], name='labels')
loss_trn = tf.nn.sigmoid_cross_entropy_with_logits(labels=ground_truth_trn, logits=output_trn)
batch_loss = tf.reduce_mean(loss_trn)

# Regularization loss
reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

# back propagation and parameter updating
train_op = tf.train.AdamOptimizer().minimize(batch_loss+reg_loss)

# other update operations
update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))

# B. Test network
data_tst = tf.placeholder(tf.float32, [None, CROP_Z, CROP_Y, CROP_X, 1], name='data')
output_tst = model(data_tst, kernel_regularizer=None, is_training=False, reuse=True)
ground_truth_tst = tf.placeholder(tf.float32, [None, 1], name='labels')
loss_tst = tf.nn.sigmoid_cross_entropy_with_logits(labels=ground_truth_tst, logits=output_tst)

################################################################################
# Training loop
################################################################################
# Now, we've built our network. Let's start a session for training
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())  # initialize the model parameters
saver = tf.train.Saver(max_to_keep=None)  # a Saver to save trained models

# build a TensorBoard summary for visualization
log_writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())
tf.summary.scalar("batch_loss", batch_loss)
log_op = tf.summary.merge_all()

def run_training():
    # Save the initial parameters
    save_path = os.path.join(SAVE_DIR, '%08d' % 0)
    saver.save(sess, save_path, write_meta_graph=False)
    print_and_log('Model saved to %s' % save_path)
    # Now, start training
    print_and_log('Training started.')
    for n_epoch in range(NUM_EPOCHES):
        num_batch = int(np.ceil(num_sample_trn / N))
        for n_batch in range(num_batch):
            idx_begin = n_batch*N
            idx_end = (n_batch+1)*N  # OK if off-the-end
            LOADED_DATA = get_random_crops(trn_image_array[idx_begin:idx_end])
            LOADED_GT = trn_label_array[idx_begin:idx_end]

            # Training step
            loss_value, _, _, summary = sess.run((batch_loss, train_op, update_op, log_op),
                {data_trn: LOADED_DATA, ground_truth_trn: LOADED_GT})

            # print to output, and save to TensorBoard
            n_iter = n_epoch*num_batch+n_batch
            print_and_log('epoch = %d, batch = %d / %d, iter = %d, loss = %f' \
                % (n_epoch, n_batch, num_batch, n_iter, loss_value))
            log_writer.add_summary(summary, n_iter)

        # save the model every SAVE_ITERS iterations (also save at the beginning)
        if (n_epoch+1) % SAVE_EPOCHES == 0:
            save_path = os.path.join(SAVE_DIR, '%08d' % (n_epoch+1))
            saver.save(sess, save_path)
            print_and_log('Model saved to %s' % save_path)

            # Test for every snapshot
            run_test('train')
            run_test('test')

def run_test(split):
    total_loss = 0
    correct = 0
    
    if split == 'train':
        print_and_log('Test on training set started.')
        all_image_array = trn_image_array
        all_label_array = trn_label_array
    elif split == 'test':
        print_and_log('Test on test set started.')
        all_image_array = tst_image_array
        all_label_array = tst_label_array
    else:
        raise ValueError('unknown data split: ' + split)
    
    num_sample = len(all_image_array)
    num_batch = int(np.ceil(num_sample / N))
    for n_batch in range(num_batch):
        idx_begin = n_batch*N
        idx_end = (n_batch+1)*N  # OK if off-the-end
        LOADED_DATA = get_random_crops(all_image_array[idx_begin:idx_end])
        LOADED_GT = all_label_array[idx_begin:idx_end]

        # Training step
        scores, losses = sess.run((output_tst, loss_tst),
                                  {data_tst: LOADED_DATA, ground_truth_tst: LOADED_GT})

        correct += np.sum((scores > 0) == LOADED_GT)
        total_loss += np.sum(losses)
        
        print_and_log('\tbatch_loss = %f' % np.mean(losses))

    avg_loss = total_loss / num_sample
    accuracy = correct / num_sample
    print_and_log('average loss on %s: %f' % (split, avg_loss))
    print_and_log('accuracy on %s: %f' % (split, accuracy))
    
if __name__ == '__main__':
    run_training()
