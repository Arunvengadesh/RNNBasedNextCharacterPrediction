#Import part**********************
import glob
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  # rnn stuff temporarily in contrib, moving back to code in TF 1.1
import os
import time
import math
import numpy as np
#import my_txtutils as txt
tf.set_random_seed(0)

import csv 
#*********************************
SEQLEN = 30
BATCHSIZE = 200
ALPHASIZE = 98
INTERNALSIZE = 512
NLAYERS = 3
learning_rate = 0.001  # fixed learning rate
dropout_pkeep = 0.8    # some dropout
epoch_len     = 50

#Load the data from Email folder 

maildir = "maildir\\allen-p\\_sent_mail/*"

def convert_from_alphabet(a):
    """Encode a character
    :param a: one character
    :return: the encoded value
    """
    if a == 9:
        return 1
    if a == 10:
        return 127 - 30  # LF
    elif 32 <= a <= 126:
        return a - 30
    else:
        return 0  # unknown

def encode_text(s):

    list = (map(lambda a: convert_from_alphabet(ord(a)), s))
    #print(list)
    
    return list     

def read_data_files(maildir, validation = False):
    
    codetext = []
    bookranges = []
    maillist = glob.glob(maildir,recursive=True)
    #print("maillist")
    #print(maillist)
    for mailfile in maillist:
        #print("mailfile")
        #print(mailfile)
        mailtext = open(mailfile, "r")
        print("Loading file "+mailfile)
        start = len(codetext)
        codetext.extend(encode_text(mailtext.read()))
        end = len(codetext)
        bookranges.append({"start": start, "end": end, "name": mailfile.rsplit("/", 1)[-1]})
        mailtext.close()  
    if len(bookranges) == 0:
        sys.exit("No training data has been found. Aborting.")
    total_len = len(codetext)
    validation_len = 0
    nb_books1 = 0 
        
    for book in reversed(bookranges):
        validation_len += book["end"]-book["start"]
        nb_books1 += 1
        if validation_len > total_len // 10:
            # print('validation_len---------->1')
            # print(validation_len)
            break
    validation_len = 0
    nb_books2 = 0
    for book in reversed(bookranges):
        validation_len += book["end"]-book["start"]
        nb_books2 += 1
        if validation_len > 90*1024:
            # print('validation_len---------->1')
            # print(validation_len)
            break

    nb_books3 = len(bookranges) // 5
    nb_books = min(nb_books1, nb_books2, nb_books3)
    if nb_books == 0 or not validation:
        cutoff = len(codetext)
    else:
        cutoff = bookranges[-nb_books]["start"]
    valitext = codetext[cutoff:]
    codetext = codetext[:cutoff]
    return codetext, valitext, bookranges


codetext, valitext, bookranges = read_data_files(maildir,validation = True)    


#print("codetext")
#print(codetext)
# display some stats on the data
epoch_size = len(codetext) // (BATCHSIZE * SEQLEN)


def print_data_stats(datalen, valilen, epoch_size):
    datalen_mb = datalen/1024.0/1024.0
    valilen_kb = valilen/1024.0
    print('datalen_mb------>')
    print(datalen_mb)
    print('valilen_kb------->{}'.format(valilen_kb))
    print(valilen_kb)
    #print("Training text size is {:.2f}MB with {:.2f}KB set aside for validation.".format(datalen_mb, valilen_kb)
    #      + " There will be {} batches per epoch".format(epoch_size))

    print("Training text size is {:.2f}MB with {:.2f}KB set aside for validation.".format(datalen_mb, valilen_kb)
          + " There will be {} batches per epoch".format(epoch_size))        

print_data_stats(len(codetext), len(valitext), epoch_size)
lr = tf.placeholder(tf.float32, name='lr')  # learning rate
batchsize = tf.placeholder(tf.int32, name='batchsize')
# inputs
X = tf.placeholder(tf.uint8, [None, None], name='X')    # [ BATCHSIZE, SEQLEN ]
Xo = tf.one_hot(X, ALPHASIZE, 1.0, 0.0)                 # [ BATCHSIZE, SEQLEN, ALPHASIZE ]
# expected outputs = same sequence shifted by 1 since we are trying to predict the next character
Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_')  # [ BATCHSIZE, SEQLEN ]
Yo_ = tf.one_hot(Y_, ALPHASIZE, 1.0, 0.0)               # [ BATCHSIZE, SEQLEN, ALPHASIZE ]
cells = [rnn.LSTMCell(INTERNALSIZE, state_is_tuple=False) for _ in range(NLAYERS)]
multicell = rnn.MultiRNNCell(cells, state_is_tuple=True)
zerostate = multicell.zero_state(BATCHSIZE, dtype=tf.float32)
Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=zerostate)
H = tf.identity(H, name='H')  # just to give it a name

Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])    # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
Ylogits = layers.linear(Yflat, ALPHASIZE)     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
Yflat_ = tf.reshape(Yo_, [-1, ALPHASIZE])     # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)  # [ BATCHSIZE x SEQLEN ]
loss = tf.reshape(loss, [batchsize, -1])      # [ BATCHSIZE, SEQLEN ]
Yo = tf.nn.softmax(Ylogits, name='Yo')        # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
Y = tf.argmax(Yo, 1)                          # [ BATCHSIZE x SEQLEN ]
Y = tf.reshape(Y, [batchsize, -1], name="Y")  # [ BATCHSIZE, SEQLEN ]
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
seqloss = tf.reduce_mean(loss, 1)
batchloss = tf.reduce_mean(seqloss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
loss_summary = tf.summary.scalar("batch_loss", batchloss)
acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
summaries = tf.summary.merge([loss_summary, acc_summary])
timestamp = str(math.trunc(time.time()))
summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training")
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1)
DISPLAY_FREQ = 50
_50_BATCHES = DISPLAY_FREQ * BATCHSIZE * SEQLEN

class Progress:
    """Text mode progress bar.
    Usage:
            p = Progress(30)
            p.step()
            p.step()
            p.step(start=True) # to restart form 0%
    The progress bar displays a new header at each restart."""
    def __init__(self, maxi, size=100, msg=""):
        """
        :param maxi: the number of steps required to reach 100%
        :param size: the number of characters taken on the screen by the progress bar
        :param msg: the message displayed in the header of the progress bat
        """
        self.maxi = maxi
        self.p = self.__start_progress(maxi)()  # () to get the iterator from the generator
        self.header_printed = False
        self.msg = msg
        self.size = size

    def step(self, reset=False):
        if reset:
            self.__init__(self.maxi, self.size, self.msg)
        if not self.header_printed:
            self.__print_header()
        next(self.p)

    def __print_header(self):
        print()
        format_string = "0%{: ^" + str(self.size - 6) + "}100%"
        print(format_string.format(self.msg))
        self.header_printed = True

    def __start_progress(self, maxi):
        def print_progress():
            # Bresenham's algorithm. Yields the number of dots printed.
            # This will always print 100 dots in max invocations.
            dx = maxi
            dy = self.size
            d = dy - dx
            for x in range(maxi):
                k = 0
                while d >= 0:
                    print('=', end="", flush=True)
                    k += 1
                    d -= dx
                d += dy
                yield k

        return print_progress



progress = Progress(DISPLAY_FREQ, size=111+2, msg="Training on next "+str(DISPLAY_FREQ)+" batches")

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
step = 0
# training loop
istate = sess.run(zerostate)  # initial zero input state (a tuple)


def rnn_minibatch_sequencer(raw_data, batch_size, sequence_size, nb_epochs):
    """
    Divides the data into batches of sequences so that all the sequences in one batch
    continue in the next batch. This is a generator that will keep returning batches
    until the input data has been seen nb_epochs times. Sequences are continued even
    between epochs, apart from one, the one corresponding to the end of raw_data.
    The remainder at the end of raw_data that does not fit in an full batch is ignored.
    :param raw_data: the training text
    :param batch_size: the size of a training minibatch
    :param sequence_size: the unroll size of the RNN
    :param nb_epochs: number of epochs to train on
    :return:
        x: one batch of training sequences
        y: on batch of target sequences, i.e. training sequences shifted by 1
        epoch: the current epoch number (starting at 0)
    """
    data = np.array(raw_data)
    data_len = data.shape[0]
    # using (data_len-1) because we must provide for the sequence shifted by 1 too
    nb_batches = (data_len - 1) // (batch_size * sequence_size)
    assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
    rounded_data_len = nb_batches * batch_size * sequence_size
    xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches * sequence_size])
    ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, nb_batches * sequence_size])

    for epoch in range(nb_epochs):
        for batch in range(nb_batches):
            x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
            y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
            x = np.roll(x, -epoch, axis=0)  # to continue the text from epoch to epoch (do not reset rnn state!)
            y = np.roll(y, -epoch, axis=0)
            yield x, y, epoch
            
def convert_to_alphabet(c, avoid_tab_and_lf=False):
    """Decode a code point
    :param c: code point
    :param avoid_tab_and_lf: if True, tab and line feed characters are replaced by '\'
    :return: decoded character
    """
    if c == 1:
        return 32 if avoid_tab_and_lf else 9  # space instead of TAB
    if c == 127 - 30:
        return 92 if avoid_tab_and_lf else 10  # \ instead of LF
    if 32 <= c + 30 <= 126:
        return c + 30
    else:
        return 0  # unknown
            
def decode_to_text(c, avoid_tab_and_lf=False):
    """Decode an encoded string.
    :param c: encoded list of code points
    :param avoid_tab_and_lf: if True, tab and line feed characters are replaced by '\'
    :return:
    """
    return "".join(map(lambda a: chr(convert_to_alphabet(a, avoid_tab_and_lf)), c))
def find_book(index, bookranges):
    return next(
        book["name"] for book in bookranges if (book["start"] <= index < book["end"]))


def print_learning_learned_comparison(X, Y, losses, bookranges, batch_loss, batch_accuracy, epoch_size, index, epoch):
    """Display utility for printing learning statistics"""
    
    print()
    # epoch_size in number of batches
    batch_size = X.shape[0]  # batch_size in number of sequences
    sequence_len = X.shape[1]  # sequence_len in number of characters
    start_index_in_epoch = index % (epoch_size * batch_size * sequence_len)
    for k in range(batch_size):
        index_in_epoch = index % (epoch_size * batch_size * sequence_len)
        decx = decode_to_text(X[k], avoid_tab_and_lf=True)
        decy = decode_to_text(Y[k], avoid_tab_and_lf=True)
        bookname = find_book(index_in_epoch, bookranges)
        formatted_bookname = "{: <10.40}".format(bookname)  # min 10 and max 40 chars
        epoch_string = "{:4d}".format(index) + " (epoch {}) ".format(epoch)
        loss_string = "loss: {:.5f}".format(losses[k])
        print_string = epoch_string + formatted_bookname + " │ {} │ {} │ {}"
        print(print_string.format(decx, decy, loss_string))
        index += sequence_len
    # box formatting characters:
    # │ \u2502
    # ─ \u2500
    # └ \u2514
    # ┘ \u2518
    # ┴ \u2534
    # ┌ \u250C
    # ┐ \u2510
    format_string = "└{:─^" + str(len(epoch_string)) + "}"
    format_string += "{:─^" + str(len(formatted_bookname)) + "}"
    format_string += "┴{:─^" + str(len(decx) + 2) + "}"
    format_string += "┴{:─^" + str(len(decy) + 2) + "}"
    format_string += "┴{:─^" + str(len(loss_string)) + "}┘"
    footer = format_string.format('INDEX', 'BOOK NAME', 'TRAINING SEQUENCE', 'PREDICTED SEQUENCE', 'LOSS')
    print(footer)
    # print statistics
    batch_index = start_index_in_epoch // (batch_size * sequence_len)
    batch_string = "batch {}/{} in epoch {},".format(batch_index, epoch_size, epoch)
    stats = "{: <28} batch loss: {:.5f}, batch accuracy: {:.5f}".format(batch_string, batch_loss, batch_accuracy)
    


    print("TRAINING STATS: {}".format(stats))



with open('SEQChangedAdamOptimizerLSTMCell_triningData.csv', 'a',newline='' ) as writeFile:
    for x, y_, epoch in rnn_minibatch_sequencer(codetext, BATCHSIZE, SEQLEN, nb_epochs=1000):

        # train on one minibatch
        feed_dict = {X: x, Y_: y_, lr: learning_rate, batchsize: BATCHSIZE}
        # This is how you add the input state to feed dictionary when state_is_tuple=True.
        # zerostate is a tuple of the placeholders for the NLAYERS=3 input states of our
        # multi-layer RNN cell. Those placeholders must be used as keys in feed_dict.
        # istate is a tuple holding the actual values of the input states (one per layer).
        # Iterate on the input state placeholders and use them as keys in the dictionary
        # to add actual input state values.
        
        for i, v in enumerate(zerostate):
            feed_dict[v] = istate[i]
        _, y, ostate, smm = sess.run([train_step, Y, H, summaries], feed_dict=feed_dict)

        # save training data for Tensorboard
        summary_writer.add_summary(smm, step)

        # display a visual validation of progress (every 50 batches)
        if step % _50_BATCHES == 0:
            feed_dict = {X: x, Y_: y_, batchsize: BATCHSIZE}  # no dropout for validation
            for i, v in enumerate(zerostate):
                feed_dict[v] = istate[i]
            y, l, bl, acc = sess.run([Y, seqloss, batchloss, accuracy], feed_dict=feed_dict)
            print_learning_learned_comparison(x[:5], y, l, bookranges, bl, acc, epoch_size, step, epoch)
        
        Batch_Acc = [[step,epoch_size,epoch,acc]]
        writer = csv.writer(writeFile)
        writer.writerows(Batch_Acc)
            # save a checkpoint (every 500 batches)
        if step // 10 % _50_BATCHES == 0:
            saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)

        # display progress bar
        progress.step(reset=step % _50_BATCHES == 0)

        # loop state around
        istate = ostate
        step += BATCHSIZE * SEQLEN
writeFile.close()
#def bat_Acc():
 #   return Batch_Acc
#BatchAcc = bat_Acc()


























