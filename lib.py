import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
from hmmlearn import hmm
from python_speech_features import mfcc
from tabulate import tabulate
import seaborn as sns
from scipy.stats import multivariate_normal

min_len = 2900
DIR = "./../recordings/"

def confusion_matrix_generator(predictions, params):
    confusion_matrix = [[0]*len(params) for i in range(len(params))]
    for item in predictions:
        predicted, actual = item
        for i, param_item in enumerate(params):
            if param_item == actual:
                for j, s_j in enumerate(params):
                    if s_j == predicted:
                        confusion_matrix[i][j] += 1

    return confusion_matrix

def lib_hmm_trainor(data, params):
    hmm_model_results = []
    test_list = []
    for p in params:
        print("training ", p)
        paths = []
        guys = []
        for d in data:
            if (d["param"] == p):
                paths.append(d["name"])
                guys.append(d["param"])
        
        x_train = np.array([])
        x_train_paths ,x_test_paths,y_train_lables,y_test_lables = train_test_split(paths,guys,test_size=0.1)
        for k in range(len(x_test_paths)):
                    test_list.append((x_test_paths[k],y_test_lables[k]))

        for j in range(len(x_train_paths)):
                    sampling_freq, audio = wavfile.read(x_train_paths[j])
                    coefs = mfcc(audio, sampling_freq ,nfft=1024)[:2900,:]
                    if j == 0:
                        x_train = coefs[:min_len,:]
                    else:
                        x_train = np.append(x_train,coefs[:min_len,:],axis = 0)
        
        model = hmm.GaussianHMM(n_components=len(params), covariance_type='diag',n_iter=12)
        model.fit(x_train)
        hmm_model_results.append(model)

    return hmm_model_results,test_list

def accuracy_evaluator(hmm_model_results,test_list, params,mode):
    accuracy = 0
    results = []
    for i in range(len(test_list)):
        path, lable = test_list[i]
        sampling_freq, audio = wavfile.read(path)
        coefs = mfcc(audio, sampling_freq ,nfft=1024)[:2900,:]
        model_prob = []
        for j in range(len(params)):
            if mode =="library hmm":
                prob = hmm_model_results[j].score(coefs)
                
            else:
                prob = hmm_model_results[j].score(coefs.T/np.amax(coefs.T))
            
            model_prob.append(prob)
        results.append((params[np.argmax(model_prob)],lable))
        if lable == params[np.argmax(model_prob)]:
            accuracy +=1

    confusion_matrix = confusion_matrix_generator(results, params)
    matrix_plotter(params, confusion_matrix)
    models_results = result_tuple_generator(params, results)
    metric_calculator(test_list, params, accuracy, models_results)

def matrix_plotter(params, confusion_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=params, yticklabels=params)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def metric_calculator(test_list, params, accu, models_results):
    metric_labels = ["Precision","Recall","F1"]
    metric_static_array = [[] for i in range(len(params))]
    macro_precision = 0
    macro_recall =0
    true_positive_sum = 0
    false_positive_sum = 0
    false_negative_sum = 0
    f1_sum = 0
    for i in range(len(params)):
        rights, false_posis ,wrongs, = models_results[params[i]]
        metric_static_array[i].append(params[i])
        # Precision formula
        if (rights+false_posis) == 0:
            metric_static_array[i].append(-1)
        else: metric_static_array[i].append(rights/(rights+false_posis))
        # Recal formula
        metric_static_array[i].append(rights/(rights+wrongs))
        # F1 formula
        if (metric_static_array[i][1] + metric_static_array[i][2]) == 0:
            metric_static_array[i].append(-1)
        else: metric_static_array[i].append((2*(metric_static_array[i][1])*(metric_static_array[i][2]))/(metric_static_array[i][1] + metric_static_array[i][2]))
        macro_recall += metric_static_array[i][2]
        macro_precision += metric_static_array[i][1]
        true_positive_sum += rights
        false_positive_sum += false_posis
        false_negative_sum += wrongs
        f1_sum += metric_static_array[i][3]

    print(tabulate(metric_static_array, metric_labels, tablefmt='fancy_grid'))
    print(f"total accuracy: {accu/len(test_list)}")
    if (true_positive_sum+false_negative_sum) == 0:
        print(f"micro precision: {true_positive_sum/(true_positive_sum + false_positive_sum)} micro recall: {-1}")
    else: print(f"micro precision: {true_positive_sum/(true_positive_sum + false_positive_sum)} micro recall: {true_positive_sum/(true_positive_sum+false_negative_sum)}")
    print(f"macro precision: {macro_precision/len(params)} macro recall: {macro_recall/len(params)}")
    print(f"average F1: {f1_sum/len(params)}")

def result_tuple_generator(params, results):
    models_results = dict()
    for digit in params:
        rights = 0
        others_predicted_this = 0
        wrongs = 0
        for res in results:
            pred,real = res
            if pred == digit and real == pred:
                rights += 1
            elif pred == digit and real != pred:
                others_predicted_this += 1
            elif real == digit and pred != real:
                wrongs +=1
        models_results[digit] = (rights,others_predicted_this,wrongs)
    
    return models_results

class HMM:
    def __init__(self, num_hidden_states):
        self.num_hidden_states = num_hidden_states
        self.rand_state = np.random.RandomState(1)

        self.initial_prob = self._normalize(self.rand_state.rand(self.num_hidden_states, 1))
        self.transition_matrix = self._stochasticize(self.rand_state.rand(self.num_hidden_states, self.num_hidden_states))

        self.mean = None
        self.covariances = None
        self.num_dimensions = None

    def _forward(self, observation_matrix):
        log_likelihood = 0.
        T = observation_matrix.shape[1]
        alpha = np.zeros(observation_matrix.shape)

        for t in range(T):
            if t == 0:
                # TODO
                alpha[:,0] = self.initial_prob[:,0] * observation_matrix[:,0]
            else:
                # TODO
                alpha[:,t] = np.matmul(self.transition_matrix.T,alpha[:,t-1]) * observation_matrix[:,t]

            alpha_sum = np.sum(alpha[:, t])
            alpha[:, t] /= alpha_sum
            log_likelihood += np.log(alpha_sum)

        return log_likelihood, alpha

    def _backward(self, observation_matrix):
        T = observation_matrix.shape[1]
        beta = np.zeros(observation_matrix.shape)

        beta[:, -1] = np.ones(observation_matrix.shape[0])

        for t in range(T - 1)[::-1]:
            # TODO
            beta[:, t] =  np.matmul(self.transition_matrix,(observation_matrix[:,t+1] * beta[:,t+1]))
            beta[:, t] /= np.sum(beta[:, t])

        return beta

    def _state_likelihood(self, obs):
        obs = np.atleast_2d(obs)
        B = np.zeros((self.num_hidden_states, obs.shape[1]))

        for s in range(self.num_hidden_states):
            np.random.seed(self.rand_state.randint(1))
            # TODO
            B[s, :] = multivariate_normal.pdf(obs.T,mean =self.mean[:,s],cov = self.covariances[:,:,s])

        return B

    def _normalize(self, x):
        return (x + (x == 0)) / np.sum(x)

    def _stochasticize(self, x):
        return (x + (x == 0)) / np.sum(x, axis=1)

    def _em_init(self, obs):
        if self.num_dimensions is None:
            self.num_dimensions = obs.shape[0]
        if self.mean is None:
            subset = self.rand_state.choice(np.arange(self.num_dimensions), size=self.num_hidden_states, replace=False)
            self.mean = obs[:, subset]
        if self.covariances is None:
            self.covariances = np.zeros((self.num_dimensions, self.num_dimensions, self.num_hidden_states))
            self.covariances += np.diag(np.diag(np.cov(obs)))[:, :, None]

        return self

    def _em_step(self, obs):
        obs = np.atleast_2d(obs)
        T = obs.shape[1]

        # TODO
        B = self._state_likelihood(obs)
        log_likelihood, alpha = self._forward(B)
        beta = self._backward(B)

        xi_sum = np.zeros((self.num_hidden_states, self.num_hidden_states))
        gamma = np.zeros((self.num_hidden_states, T))

        for t in range(T - 1):
            # TODO
            partial_sum = np.matmul(alpha[:,t],(beta[:,t+1].T*B[:,t+1].T)) * self.transition_matrix
            xi_sum += self._normalize(partial_sum)
            # TODO
            partial_g = alpha[:,t] * beta[:,t]
            gamma[:, t] = self._normalize(partial_g)

        # TODO
        partial_g = alpha[:, T - 1] * beta[:, T - 1]
        gamma[:, -1] = self._normalize(partial_g)

        # TODO
        expected_prior = np.reshape(gamma[:, 0],(-1,1))
        expected_transition = self._stochasticize(xi_sum/np.sum(xi_sum,axis=(0,1)))

        expected_covariances = np.zeros((self.num_dimensions, self.num_dimensions, self.num_hidden_states))
        expected_covariances += .01 * np.eye(self.num_dimensions)[:, :, None]

        gamma_state_sum = np.sum(gamma, axis=1)
        gamma_state_sum = gamma_state_sum + (gamma_state_sum == 0)

        expected_mean = np.zeros((self.num_dimensions, self.num_hidden_states))
        for s in range(self.num_hidden_states):
            gamma_obs = obs * gamma[s, :]
            expected_mean[:, s] = np.sum(gamma_obs, axis=1) / gamma_state_sum[s]

        self.initial_prob = expected_prior
        self.mean = expected_mean
        self.covariances = expected_covariances
        self.transition_matrix = expected_transition

        return log_likelihood

    def train(self, obs, num_iterations=1080):
        for i in range(num_iterations):
            self._em_init(obs)
            self._em_step(obs)
        return self

    def score(self, obs):
        B = self._state_likelihood(obs)
        log_likelihood, _ = self._forward(B)
        return log_likelihood

def custom_hmm_trainor(data, params):
    hmm_model_results = []
    test_list = []
    for p in params:
        print("training ", p)
        paths = []
        guys = []
        for d in data:
            if (d["param"] == p):
                paths.append(d["name"])
                guys.append(d["param"])

        x_train = np.array([])
        x_train_paths, x_test_paths, _, y_test_lables = train_test_split(paths,guys,test_size=0.1)
        for k in range(len(x_test_paths)):
                    test_list.append((x_test_paths[k],y_test_lables[k]))

        for j in range(len(x_train_paths)):
                    sampling_freq, audio = wavfile.read(x_train_paths[j])
                    mfccs = mfcc(audio, sampling_freq ,nfft=1024)[:2900,:]
                    if j == 0:
                        x_train = mfccs[:min_len,:]
                    else:
                        x_train = np.append(x_train,mfccs[:min_len,:],axis = 0)

        model = HMM(len(params))
        model.train(x_train.T/np.amax(x_train.T),2)
        hmm_model_results.append(model)

    return hmm_model_results,test_list