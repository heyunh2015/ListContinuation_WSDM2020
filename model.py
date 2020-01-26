from modules import *


class Model():
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.creator = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            creator_emb, creator_emb_table = embedding(self.creator,
                                                 vocab_size=usernum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="creator_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            self.seq += t

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            
            self.seq *= mask #YunHe: actually this is important to block paddings, so that 
            #gating_seq *= mask
            # Build blocks

            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):

                    # CPPM
                    normalized_seq = normalize(self.seq)
                    seq_CPPM = multihead_attention(queries=normalized_seq,
                                                   keys=self.seq,
                                                   num_units=args.hidden_units,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")#, coherences

                    global_vector = tf.get_variable('global_vector', dtype=tf.float32, shape=[args.hidden_units], regularizer=tf.contrib.layers.l2_regularizer(0.1))
                    print(global_vector.get_shape().as_list)
                    global_vector = tf.tile(tf.expand_dims(global_vector, 0), [tf.shape(self.input_seq)[1], 1])
                    print(global_vector.get_shape().as_list)
                    global_vector = tf.tile(tf.expand_dims(global_vector, 0), [tf.shape(self.input_seq)[0], 1, 1])
                    print(global_vector.get_shape().as_list)

                    # GUPM
                    seq_GUPM = multihead_attention_vanilla(queries=global_vector,#normalize(self.seq),
                                                    keys=self.seq,
                                                    num_units=args.hidden_units,
                                                    num_heads=args.num_heads,
                                                    dropout_rate=args.dropout_rate,
                                                    is_training=self.is_training,
                                                    causality=True,
                                                    scope="vanilla_attention")

                    # Gating Network
                    gating_vector = tf.get_variable('gating_vector', dtype=tf.float32, shape=[args.hidden_units], regularizer=tf.contrib.layers.l2_regularizer(0.1))
                    print(gating_vector.get_shape().as_list)
                    gating_vector = tf.tile(tf.expand_dims(gating_vector, 0), [tf.shape(self.input_seq)[1], 1])
                    print(gating_vector.get_shape().as_list)
                    gating_vector = tf.tile(tf.expand_dims(gating_vector, 0), [tf.shape(self.input_seq)[0], 1, 1])
                    print(gating_vector.get_shape().as_list)

                    List_embedding_based_input = multihead_attention_vanilla(gating_vector,#normalize(self.seq),
                                                    keys=self.seq,
                                                    num_units=args.hidden_units,
                                                    num_heads=args.num_heads,
                                                    dropout_rate=args.dropout_rate,
                                                    is_training=self.is_training,
                                                    causality=True,
                                                    scope="gating_network")

                    Consistency_based_input = centroid_coherence_func(inputs = self.seq, num_units=args.hidden_units, scope='distanceBetweenLastItemWithCentroid')

                    seq_gating_input = tf.concat((List_embedding_based_input, Consistency_based_input), -1)#, range_coherence, creator_emb
                    
                    seq_gating_input = tf.layers.dropout(seq_gating_input, rate=args.dropout_rate, training=tf.convert_to_tensor(self.is_training))  
                    weights = tf.layers.dense(seq_gating_input, 3, activation=None)
                    weights = tf.nn.softmax(weights)
                    print(weights.get_shape().as_list)
                    weights_self = tf.expand_dims(weights[:, :, 0], -1)
                    weights_vanilla = tf.expand_dims(weights[:, :, 1], -1)
                    weights_last = tf.expand_dims(weights[:, :, 2], -1)
                    self.seq = seq_CPPM*weights_self + seq_GUPM*weights_vanilla +  normalized_seq*weights_last#

                    # Add user (creator) embedding 
                    creator_emb = tf.tile(tf.expand_dims(creator_emb, 1), [1, tf.shape(self.input_seq)[1], 1])
                    creator_emb = tf.layers.dropout(creator_emb,
                                                     rate=args.dropout_rate,
                                                     training=tf.convert_to_tensor(self.is_training))
                    self.seq += creator_emb
                
                    self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                          dropout_rate=args.dropout_rate, is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq)

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])

        self.test_item = tf.placeholder(tf.int32, shape=(101))
        test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, 101])
        self.test_logits = self.test_logits[:, -1, :]

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        # self.loss = tf.reduce_sum(
        #     - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
        #     tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        # ) / tf.reduce_sum(istarget)
        self.loss = tf.reduce_sum(- tf.log(tf.sigmoid(self.pos_logits-self.neg_logits) + 1e-24) * istarget)/ tf.reduce_sum(istarget)
        #self.loss = tf.reduce_sum(tf.sigmoid(self.neg_logits-self.pos_logits)  * istarget + tf.sigmoid(tf.square(self.neg_logits))* istarget)/ tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, creator, seq, item_idx):
        return sess.run(self.test_logits,
                        {self.creator: creator, self.input_seq: seq, self.test_item: item_idx, self.is_training: False})
