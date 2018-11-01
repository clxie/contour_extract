#!/bin/env python

class Train():
    def __init__(self):
        self.net=Network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self,datadir):
        batch_size=50
        train_step=20000
        step=0
        save_interval=500
        saver = tf.train.Saver()

        merged_summary_op = tf.summary.merge_all()
        merged_writer = tf.summary_FileWriter('./log',tf.sess.graph)

        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            step = self.sess.run(self.net.global_step)
        
        while step < train_step:
            x1,label1 = read-data(datadir)
            _,merged_summary = self.sess.run([self.net.train, merged_summary_op],
                                    feed_dict={self.net.x:x1,self.net.label:label1})

        if step % save_interval == 0:
            saver.save(self.sess,CKPT_DIR + '/model', global_step=step)
            merged_writer.add_summary(merged_summary, step)
