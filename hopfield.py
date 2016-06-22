# Author: Murat Kirtay, The BioRobotics Inst./SSSA/
# Date: 22/06/2016
# Description: Solo main for testing hopfield network

import hn
import cv2
import numpy as np

def main():

    # paths for training and testing paths
    training_patterns = 'patterns/training/'
    testing_patterns = 'patterns/noise/5p'


    # num_of_neurons = num_of_col * num_of_rows
    num_of_cols, num_of_rows = 20, 20
    neurons = num_of_cols * num_of_rows
    size = (num_of_cols, num_of_rows)

    unchanged_states = 400

    inputs = hn.create_file_names(training_patterns)
    binaries = hn.perform_bin_image_processing(inputs, size)
    test_inputs = hn.create_file_names(testing_patterns)

    test_binp = hn.binarize_pattern(test_inputs[1])
    #prev_test_binp = np.copy(test_binp)
    prev_test_binp = np.zeros((1, neurons), dtype='int64')
    np.copyto(prev_test_binp, test_binp)

    # construct weight mat with binary values
    total_w = hn.construct_weight_mat(binaries, neurons)


    print "---------Testing Pattern--------------"
    print hn.format_patterns(test_binp, num_of_cols)

    # return (test_bin, prev_test, flipped_bits, state_counter)
    res = hn.run_hopfield(test_binp, prev_test_binp, total_w, neurons)

    test_b, flip_bits, state_cntr = res[0], res[1], res[2]

    print "---------Converged Pattern--------------"
    print hn.format_patterns(test_b, num_of_cols)

    conv_rate = hn.extract_convergence_rate(test_b, prev_test_binp, neurons)
        
    info = (conv_rate, flip_bits, state_cntr, unchanged_states)
    hn.print_hn_info(info)

    # display as an image
    conv_test_img =hn. convert_pattern_to_image(test_b, size)

    rsized = cv2.resize(conv_test_img, (63,79))
    cv2.imwrite("converged.png", rsized)

    cv2.imshow("Converged Pattern", rsized)

    cv2.waitKey(0)

if __name__ == '__main__':
    main()
