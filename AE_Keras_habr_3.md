
# Автоэнкодеры в Keras

# Часть 3: Вариационные автоэнкодеры (VAE)

### Содержание

* Часть 1: Введение
* Часть 2: *Manifold learning* и скрытые (*latent*) переменные
* **Часть 3: Вариационные автоэнкодеры (*VAE*)**
* Часть 4: *Conditional VAE*
* Часть 5: *GAN* (Generative Adversarial Networks) и tensorflow
* Часть 6: *VAE* + *GAN*



В прошлой части мы уже обсуждали, что такое скрытые переменные, взглянули на их распределение, а так же поняли, что из распределения скрытых переменных в обычных автоэнкодерах сложно генерировать новые объекты. Для того, чтобы можно было генерировать новые объекты, пространство *скрытых переменных* (*latent variables*) должно быть предсказуемым. 

Имея какое-то одно распределение $Z$ можно получить произвольное другое $X = g(Z)$, например,

пусть $Z$ - обычное нормальное распределение, $g(Z) = \frac{Z}{|Z|}+ \frac{Z}{10}$ - тоже случайное распределение, но выглядит совсем по другому

Код: (скрыто)


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


Z = np.random.randn(150, 2)
X = Z/(np.sqrt(np.sum(Z*Z, axis=1))[:, None]) + Z/10

fig, axs = plt.subplots(1, 2, sharex=False, figsize=(16,8))

ax = axs[0]
ax.scatter(Z[:,0], Z[:,1])
ax.grid(True)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

ax = axs[1]
ax.scatter(X[:,0], X[:,1])
ax.grid(True)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
```

![](https://habrastorage.org/web/fac/4db/fc6/fac4dbfc6dac4e14a89b4eea7607af99.png)

Пример выше из ***[1]***

Таким образом, если подобрать правильные функции, то можно отобразить пространства скрытых переменных обычных автоэнкодеров в какие-то хорошие пространства, например такие, где распределение нормально. А потом обратно.

С другой стороны специально учиться отображать одни скрытые пространства в другие вовсе не обязательно. Если есть какие-то полезные скрытые пространства, то правильный автоэнкодер научится им по пути сам, но отображать, в конечно итоге, будет в нужное нам пространство.

***Вариационные автоэнкодеры*** (*Variational Autoencoders*) - это автоэнкодеры, которые учатся отображать объекты в заданное скрытое пространство и, соответственно, сэмплить из него. Поэтому *вариационные автоэнкодеры* относят так же к семейству генеративных моделей.

![](https://habrastorage.org/web/725/94b/5de/72594b5de85e4e58a0ae071bf2ab2ca7.png)

Иллюстрация из ***[2]***

Далее выжимка теории из ***[1]*** лежащая в основе *VAE*.

Пусть $Z$ - скрытые переменные, а $X$ - данные. 
На примере нарисованых цифр рассмотрим естесственный генеративный процесс, который сгенерировал нашу выборку:
$$
P(X) = \int_{z} P(X|Z)P(Z)dZ
$$

- $P(X)$ вероятностное распределение изображений цифр на картинках, т.е. вероятность конкретного изображения цифры впринципе быть нарисованым (если картинка не похожа на цифру, то эта вероятность крайне мала, и наоборот),
- $P(Z)$ - вероятностное распределение скрытых факторов, например, распределение толщины штриха,
- $P(X|Z)$ - распределение вероятности картинок при заданных скрытых факторах, одни и те же факторы могут привезти к разным картинкам (один и тот же человек в одних и тех же условиях не рисует абсолютно одинаковые цифры)

Представим $P(X|Z)$ как сумму некоторой генерирующей функции $f(Z)$ и некоторого сложного шума $\epsilon$

$$
P(X|Z) = f(Z) + \epsilon
$$

Мы хотим построить некоторый искусственный генеративный процесс, который будет создавать объекты близкие в некоторой метрике к тренировачным $X$.

$$
P(X;\theta) = \int_{z} P(X|Z;\theta)P(Z)dZ \ \ \ (1)
$$
и снова
$$
P(X|Z;\theta) = f(Z;\theta) + \epsilon
$$

$f(Z;\theta)$ - некоторое семейсто функций, которое представляет наша модель, а $\theta$ - ее параметры. Выбирая метрику - мы выбираем то, какого вида нам представляется шум $\epsilon$. Если метрика $L_2$, то мы считаем шум нормальным и тогда:

$$
P(X|Z;\theta) = N(X|f(Z;\theta), \sigma^2 I),
$$

По принципу максимального правдоподобия нам остается оптимизировать параметры $\theta$ для того, чтобы максимизировать $P(X)$, т.е. вероятность появления объектов из выборки.

Проблема в том, что оптимизировтаь интеграл (1) напрямую мы не можем: пространство может быть высокоразмерное, объектов много, да и метрика плохая. С другой стороны, если задуматься, то к каждому конкретному $X$ может привезти лишь очень небольшое подмножество $Z$, для остальных же $P(X|Z)$ будет очень близок к нулю. 
И при оптимизации достаточно сэмплить только из хороших $Z$.

Для того чтобы знать из каких $Z$ нам надо сэмплить, введем новое распределение $Q(Z|X)$, которое в зависимости от $X$ будет показывать распределение $Z \sim Q$, которое могло привезти к этому $X$.

Запишем сперва расстояние Кульбака-Лейблера (несимметричная мера "похожести" двух распределений, подробнее ***[3]***) между
$Q(Z|X)$ и реальным $P(Z|X)$:

$$
KL[Q(Z|X)||P(Z|X)] = \mathbb{E}_{Z \sim Q}[\log Q(Z|X) - \log P(Z|X)]
$$

Применяем формулу Байеса:

$$
KL[Q(Z|X)||P(Z|X)] = \mathbb{E}_{Z \sim Q}[\log Q(Z|X) - \log P(X|Z) - \log P(Z)] + \log P(X)
$$

Выделяем еще одно расстояние Кульбака-Лейблера:

$$
KL[Q(Z|X)||P(Z|X)] = KL[Q(Z|X)||\log P(Z)] - \mathbb{E}_{Z \sim Q}[\log P(X|Z)] + \log P(X)
$$

В итоге получаем тождество:

$$
\log P(X) - KL[Q(Z|X)||P(Z|X)] = \mathbb{E}_{Z \sim Q}[\log P(X|Z)] - KL[Q(Z|X)||P(Z)]
$$


Это тождество - краеугольный камень *вариационных автоэнкодеров*, оно верно для любых $Q(Z|X)$ и $P(X,Z)$.

Пусть $Q(Z|X)$ и $P(X|Z)$ зависят от параметров: $Q(Z|X;\theta_1)$ и $P(X|Z;\theta_2)$, а $P(Z)$ - нормальное $N(0,I)$, тогда получаем:

$$
\log P(X;\theta_2) - KL[Q(Z|X;\theta_1)||P(Z|X;\theta_2)] = \mathbb{E}_{Z \sim Q}[\log P(X|Z;\theta_2)] - KL[Q(Z|X;\theta_1)||N(0,I)]
$$


Взглянем повнимательнее на то, что у нас получилось:
- во-первых, $Q(Z|X;\theta_1)$, $P(X|Z;\theta_2)$ подозрительно похожи на энкодер и декодер (точнее декодер это $f$ в выражении $P(X|Z;\theta_2) = f(Z;\theta_2) + \epsilon$)
- слева в тождестве - значение, которое мы хотим максимизировать для элементов нашей тренировачной выборки $X$ + некоторая ошибка $KL$ ($KL(x,y) \ge 0 \ \ \forall x,y$), которая, будем надеяться, при достаточной емкости $Q$ уйдет в 0,
- справа значение, которое мы можем оптимизировать градиентным спуском, где первый член имеет смысл качества предсказания $X$ декодером по значениям $Z$, а второй член, это расстояние К-Л между распределением $Z \sim Q$, которое предсказывает энкодер для конкретного $X$, и распределением $Z$ для всех $X$ сразу

Для того, чтобы иметь возможность оптимизировать правую часть градиентным спуском, осталось разобраться с двумя вещами:

#### 1. Точнее определим что такое $Q(Z|X;\theta_1)$
Обычно $Q$ выбирается нормальным распределением:

$$
Q(Z|X;\theta_1) = N(\mu(X;\theta_1), \Sigma(X;\theta_1))
$$
То есть энкодер для каждого $X$ предсказывает 2 значения: среднее $\mu$ и вариацию $\Sigma$ нормального распределения, из которого уже сэмплируются значения. Работает это все примерно вот так:  
![](https://habrastorage.org/web/3df/aaf/bca/3dfaafbca9924d3187da7a7a9367fe93.png)

Иллюстрация из ***[2]***


При том, что для каждой отдельной точки данных $X$ энкодер предсказывает некоторое нормальное распределение $P(Z|X) = N(\mu(X), \Sigma(X))$, для априорного распределения $Х$: $P(Z) = N(0, I)$, что получается из формулы, и это потрясающе.
![](https://habrastorage.org/web/f31/674/a0b/f31674a0bb974f11983fac5a8ce1cedf.png)

Иллюстрация из ***[2]***


При этом $KL[Q(Z|X;\theta_1)||N(0,I)]$ принимает вид:

$$
KL[Q(Z|X;\theta_1)||N(0,I)] = \frac{1}{2}\left(tr(\Sigma(X)) + \mu(X)^T\mu(X) - k - \log \det \Sigma(X) \right)
$$

#### 2. Разберемся с тем, как распространять ошибки через $\mathbb{E}_{Z \sim Q}[\log P(X|Z;\theta_2)]$
Дело в том, что здесь мы берем случайные значения $Z \sim Q(Z|X;\theta_1)$ и передаем их в декодер.
Ясно, что распросранять ошибки через случайные значения напрямую нельзя, поэтому используется так называемый *трюк с репараметризацией* (*reparametrization trick*).

Схема получается вот такая:
![](https://habrastorage.org/web/a4e/ec5/3a3/a4eec53a3cf24b289e494e4f03f71a39.png)

Иллюстрация из ***[1]***

Здесь на левой картинке схема без трюка, а на правой с трюком.
Красным цветом показано семплирование, а синим вычисление ошибки.  
То есть по сути просто берем предсказанное энкодером стандартное отклонение $\Sigma$ умножаем на случайное число из $N(0,I)$ и добавляем предсказанное среднее $\mu$.

Прямое растространение на обеих схемах абсолютно одинаковое, однако на правой схеме работает обратное распространение ошибки.

После того, как мы обучили такой вариационный автоэнкодер, декодер становится полноправной генеративной моделью. По сути и энкодер то нужен в основном для того, чтобы обучить декодер отдельно быть генеративной моделью. 
![](https://habrastorage.org/web/d1d/500/61b/d1d50061b0dd4836af3bd9d127b0a7e2.png)

Иллюстрация из ***[2]***

<img src="https://habrastorage.org/web/211/52c/3b3/21152c3b383c45efb5819e5358da522b.png", width=300/>

Иллюстрация из ***[1]***

Но то, что энкодер и декодер вместо образуют еще и полноценный автоэнкодер, это очень приятный плюс.

# VAE в Keras 

Теперь, когда мы разобрались в том, что такое вариационные автоэнкодеры, напишем такой на *Keras*

Импортируем необходимые библиотеки и датасет


```python
import sys
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test  = x_test .astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test  = np.reshape(x_test,  (len(x_test),  28, 28, 1))
```

Зададим основные параметры. Скрытое пространство возьмем размерности 2, чтобы позже генерировать из него и визуализировать результат.  
***Замечание***: размерность 2 крайне мала, особенно в метрике $L_2$, поэтому следует ожидать, что цифры получатся очень размытыми.


```python
batch_size = 500
latent_dim = 2
dropout_rate = 0.3
start_lr = 0.0001
```

Напишем модели вариационного автоэнкодера. 

Для того, чтобы обучение происходило быстрее и более качественно, добавим слои *dropout* и *batch normalization*.
А в декодере используем в качестве активации *leaky ReLU*, которую добавляем отдельным слоем после *dense* слоев без активации.

Функция *sampling* реализует сэмплирование значений $Z$ из $Q(X)$ с использованием трюка репараметризации.

*vae_loss* это правая часть из уравнения:

$$
\log P(X;\theta_2) - KL[Q(Z|X;\theta_1)||P(Z|X;\theta_2)] = \mathbb{E}_{Z \sim Q}[\log P(X|Z;\theta_2)] - \left(\frac{1}{2}\left(tr(\Sigma(X)) + \mu(X)^T\mu(X) - k - \log \det \Sigma(X) \right)\right)
$$

далее будет использоваться в качестве лосса.


```python
from keras.layers import Input, Dense 
from keras.layers import BatchNormalization, Dropout, Flatten, Reshape, Lambda
from keras.models import Model

from keras.objectives import binary_crossentropy
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K


def create_vae():
    models = {}

    # Добавим Dropout и BatchNormalization
    def apply_bn_and_dropout(x):
        return Dropout(dropout_rate)(BatchNormalization()(x))

    # Энкодер
    input_img = Input(batch_shape=(batch_size, 28, 28, 1))
    x = Flatten()(input_img)
    x = Dense(256, activation='relu')(x)
    x = apply_bn_and_dropout(x)
    x = Dense(128, activation='relu')(x)
    x = apply_bn_and_dropout(x)

    # Предсказываем параметры распределений
    # Вместо того, чтобы предсказывать стандартное отклонение, предсказываем логарифм вариации
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    # Сэмплирование из Q с трюком репараметризации
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    l = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    models["encoder"]  = Model(input_img, l, 'Encoder') 
    models["z_meaner"] = Model(input_img, z_mean, 'Enc_z_mean')
    models["z_lvarer"] = Model(input_img, z_log_var, 'Enc_z_log_var')

    # Декодер
    z = Input(shape=(latent_dim, ))
    x = Dense(128)(z)
    x = LeakyReLU()(x)
    x = apply_bn_and_dropout(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = apply_bn_and_dropout(x)
    x = Dense(28*28, activation='sigmoid')(x)
    decoded = Reshape((28, 28, 1))(x)

    models["decoder"] = Model(z, decoded, name='Decoder')
    models["vae"]     = Model(input_img, models["decoder"](models["encoder"](input_img)), name="VAE")

    def vae_loss(x, decoded):
        x = K.reshape(x, shape=(batch_size, 28*28))
        decoded = K.reshape(decoded, shape=(batch_size, 28*28))
        xent_loss = 28*28*binary_crossentropy(x, decoded)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return (xent_loss + kl_loss)/2/28/28

    return models, vae_loss

models, vae_loss = create_vae()
vae = models["vae"]
```

***Замечание***: мы использовали *Lambda*-слой с функцией сэмплирующей из $N(0, I)$ из нижележащего фреймворка, которая явно требует размер батча. Во всех моделях, в которых присутствует этот слой мы теперь вынуждены передавать именно такой размер батча (то есть в "encoder" и "vae*).

Функцией оптимизации возьмем *Adam* или *RMSprop*, обе показывают хорошие результаты.  


```python
from keras.optimizers import Adam, RMSprop

vae.compile(optimizer=Adam(start_lr), loss=vae_loss)
```

Код рисования рядов цифр и цифр из многообразия


```python
digit_size = 28

def plot_digits(*args, invert_colors=False):
    args = [x.squeeze() for x in args]
    n = min([x.shape[0] for x in args])
    figure = np.zeros((digit_size * len(args), digit_size * n))

    for i in range(n):
        for j in range(len(args)):
            figure[j * digit_size: (j + 1) * digit_size,
                   i * digit_size: (i + 1) * digit_size] = args[j][i].squeeze()

    if invert_colors:
        figure = 1-figure

    plt.figure(figsize=(2*n, 2*len(args)))
    plt.imshow(figure, cmap='Greys_r')
    plt.grid(False)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()


n = 15 # Картинка с 15x15 цифр
digit_size = 28

from scipy.stats import norm
# Так как сэмплируем из N(0, I), то сетку узлов, в которых генерируем цифры берем из обратной функции распределения
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

def draw_manifold(generator, show=True):
    # Рисование цифр из многообразия
    figure = np.zeros((digit_size * n, digit_size * n))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.zeros((1, latent_dim))
            z_sample[:, :2] = np.array([[xi, yi]])

            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].squeeze()
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit
    if show:
        # Визуализация
        plt.figure(figsize=(15, 15))
        plt.imshow(figure, cmap='Greys_r')
        plt.grid(None)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)        
        plt.show()
    return figure
```

Часто в процессе обучения модели требуется выполнять какие-то действия: изменять *learning_rate*, сохранять промежуточные результаты, сохранять модель, рисовать картинки и т.д.

Для этого в *keras* есть коллбэки, которые передаются в метод *fit* перед началом обучения. Например, чтобы влиять на *learning rate* в процессе обучения есть такие коллбэки, как *LearningRateScheduler*, *ReduceLROnPlateau*, чтобы сохранять модель - *ModelCheckpoint*.

Отдельный коллбэк нужен для того, чтобы следить за процессом обучения в *TensorBoard*. Он автоматически будет добавлять в файл логов все метрики и лоссы, которые считаются между эпохами.

Для случая, когда требуется выполнения произвольных функций в процессе обучения, существует *LambdaCallback*. Он запускает выполнение произвольных функций в заданные моменты обучения, например между эпохами или батчами.  
Будем следить за процессом обучения, изучая, как генерируются цифры из $N(0, I)$.


```python
from IPython.display import clear_output
from keras.callbacks import LambdaCallback, ReduceLROnPlateau, TensorBoard

# Массивы в которые будем сохранять результаты, для последующей визуализации
figs = []
latent_distrs = []
epochs = []

# Эпохи в которые будем сохранять
save_epochs = set(list((np.arange(0, 59)**1.701).astype(np.int)) + list(range(10)))

# Отслеживать будем на вот этих цифрах
imgs = x_test[:batch_size]
n_compare = 10

# Модели
generator      = models["decoder"]
encoder_mean   = models["z_meaner"]


# Фунция, которую будем запускать после каждой эпохи
def on_epoch_end(epoch, logs):
    if epoch in save_epochs:
        clear_output() # Не захламляем output

        # Сравнение реальных и декодированных цифр
        decoded = vae.predict(imgs, batch_size=batch_size)
        plot_digits(imgs[:n_compare], decoded[:n_compare])

        # Рисование многообразия
        figure = draw_manifold(generator, show=True)

        # Сохранение многообразия и распределения z для создания анимации после
        epochs.append(epoch)
        figs.append(figure)
        latent_distrs.append(encoder_mean.predict(x_test, batch_size))

        
# Коллбэки
pltfig = LambdaCallback(on_epoch_end=on_epoch_end)
# lr_red = ReduceLROnPlateau(factor=0.1, patience=25)
tb     = TensorBoard(log_dir='./logs')


# Запуск обучения 
vae.fit(x_train, x_train, shuffle=True, epochs=1000,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        callbacks=[pltfig, tb],
        verbose=1)
```

Теперь, если установлен *TensorBoard*, можно следить за процессом обучения.

Вот как этот энкодер восстанавливает изображения:
![](https://habrastorage.org/web/670/bc6/c44/670bc6c44122449eb1d363144736f40f.png)

А вот результат сэмплирования из $N(0|I)$
<img src="https://habrastorage.org/web/92a/ac4/61b/92aac461bd794121874d9b307448ad2f.png" width="600"/>

Вот так выглядит процесс обучения генерации цифр (скрыто):
<img src="https://habrastorage.org/web/fe5/68a/a5e/fe568aa5ee8e4d939ec067584fc34874.gif" width="600"/>

Распределение кодов в скрытом пространстве (скрыто).  
<img src="https://habrastorage.org/web/c18/f23/cd8/c18f23cd8e044486a1336dd40551a5bc.gif" width="600"/>

Не идеально нормальное, но довольно близко (особенно, учитывая, что размерность скрытого пространства всего 2).

Кривая обучения в *TensorBoard*
<img src="https://habrastorage.org/web/320/18b/d03/32018bd03465439e9eed0b418f1067bf.png" width="800"/>

### Код создания гифок


```python
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import matplotlib

def make_2d_figs_gif(figs, epochs, fname, fig): 
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=False)
    im = plt.imshow(np.zeros((28,28)), cmap='Greys_r', norm=norm)
    plt.grid(None)
    plt.title("Epoch: " + str(epochs[0]))

    def update(i):
        im.set_array(figs[i])
        im.axes.set_title("Epoch: " + str(epochs[i]))
        im.axes.get_xaxis().set_visible(False)
        im.axes.get_yaxis().set_visible(False)
        return im
    
    anim = FuncAnimation(fig, update, frames=range(len(figs)), interval=100)
    anim.save(fname, dpi=80, writer='imagemagick')

def make_2d_scatter_gif(zs, epochs, c, fname, fig):
    im = plt.scatter(zs[0][:, 0], zs[0][:, 1], c=c, cmap=cm.coolwarm)
    plt.colorbar()
    plt.title("Epoch: " + str(epochs[0]))

    def update(i):
        fig.clear()
        im = plt.scatter(zs[i][:, 0], zs[i][:, 1], c=c, cmap=cm.coolwarm)
        im.axes.set_title("Epoch: " + str(epochs[i]))
        im.axes.set_xlim(-5, 5)
        im.axes.set_ylim(-5, 5)
        return im

    anim = FuncAnimation(fig, update, frames=range(len(zs)), interval=150)
    anim.save(fname, dpi=80, writer='imagemagick')
    
make_2d_figs_gif(figs, epochs, "./figs3/manifold.gif", plt.figure(figsize=(10,10)))
make_2d_scatter_gif(latent_distrs, epochs, y_test, "./figs3/z_distr.gif", plt.figure(figsize=(10,10)))
```

В следующей части посмотрим как генерировать цифры нужного лейбла, а также как переносить стиль с одной цифры на другую.

## Полезные ссылки и литература

Теоретическая часть основана на статье:  
[1] Tutorial on Variational Autoencoders, Carl Doersch, 2016, <https://arxiv.org/abs/1606.05908>
и фактически является ее кратким изложением

Многие картинки взяты из блога Isaac Dykeman:  
[2] Isaac Dykeman, <http://ijdykeman.github.io/ml/2016/12/21/cvae.html>  

Подробнее прочитать про расстояние Кульбака-Лейблера на русском можно здесь   
[3] <http://www.machinelearning.ru/wiki/images/d/d0/BMMO11_6.pdf>  

Код частично основан на статье *Francois Chollet*:  
[4] <https://blog.keras.io/building-autoencoders-in-keras.html>

Другие интересные ссылки:  
<http://blog.fastforwardlabs.com/2016/08/12/introducing-variational-autoencoders-in-prose-and.html>  
<http://kvfrans.com/variational-autoencoders-explained/>
