
import seaborn as sns
from keras.utils import to_categorical, plot_model
from keras.optimizers import SGD, Adam

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from cnn import Convolucao
import datetime
import time

def getDateStr(): # funcao para adicionar o horario do treinamento no arquivo treinado
        return str('{date:%d_%m_%Y_%H_%M}').format(date=datetime.datetime.now())

def getTimeMin(start, end): # funcao para mostrar o tempo total de treinamento
        return (end - start)/60

EPOCHS = 30 #a cada epoch você realiza o treino com as imagens
CLASS = 10
FILE_NAME = 'cnn_model_LIBRAS_'

print("\n\n ----------------------INICIO --------------------------\n")
print('[INFO] [INICIO]: ' + getDateStr()) # vai mostrar o horario inicial de treino obtido na funcao acima
print('[INFO] Download dataset usando keras.preprocessing.image.ImageDataGenerator')


train_datagen = ImageDataGenerator( #guarda as imagens geradas
        rescale=1./255, #rescale: fator de reescalonamento. Por padrão é None. Se o valor de reescalar for zero ou nenhum, então nenhum reescalonamento é aplicado, caso contrário, multiplicamos os dados pelo valor fornecido
        shear_range=0.2, #shear_range: Intensidade de cisalhamento (ângulo de cisalhamento no sentido anti-horário em graus)
        zoom_range=0.2, #zoom_range: Alcance para zoom aleatório
        horizontal_flip=True, #Boolean. Inverta aleatoriamente as entradas horizontalmente..
        validation_split=0.25) #validation_split: Uma fração das imagens são reservadas para validação. Esse valor é restrito de 0 a 1.



test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.05) #guarda os testes

# produz base de dados de treino gerada pelo modelo #
training_set = train_datagen.flow_from_directory(
        './dataset/treinamento_palavra', #a pasta com as imagens originais, o keras define as pastas de cada letra como classe para o treinamento.
        target_size=(64, 64),#dimensoes das imagens
        color_mode = 'rgb',#canais de cores
        batch_size=32,#quantidade de imagens treinadas por epoch
        shuffle=False,#shuffle: se os dados devem ser embaralhados (padrão: True) Se definido como False, classifica os dados em ordem alfanumérica.
        class_mode='categorical') #classifica o treinamento por categorias

#Pega o caminho para o diretório e gera lotes de dados aumentados.
test_set = test_datagen.flow_from_directory(
        './dataset/teste_palavra', #diretório: string, caminho para o diretório de destino. Deve conter um subdiretório por classe. Quaisquer imagens PNG, JPG, BMP, PPM ou TIF dentro de cada um dos subdiretórios da árvore de diretórios serão incluídos no gerador. Veja este script para mais detalhes.
        target_size=(64, 64),
        color_mode = 'rgb',
        batch_size=32,
        shuffle=False, #shuffle: se os dados devem ser embaralhados (padrão: True) Se definido como False, classifica os dados em ordem alfanumérica.
        class_mode='categorical')

# inicializar e otimizar modelo
print("[INFO] Inicializando e otimizando a CNN...")
start = time.time()

early_stopping_monitor = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)


model = Convolucao.build(64, 64, 3, CLASS) #criação da rede neural


model.compile(optimizer=SGD(0.01), loss="categorical_crossentropy",
              metrics=["accuracy"])

'''
Ao definir verbose 0, 1 ou 2, você apenas diz como deseja 'ver' o progresso do treinamento para cada época.

verbose=0 não mostrará nada (silencioso)
verbose=1 mostrará uma barra de progresso animada
verbose=2 apenas mencionará o número de época
'''
# treinar a CNN
print("[INFO] Treinando a CNN...")
classifier = model.fit(
        training_set,
        #Número total de etapas (lotes de amostras) para produzir do gerador antes de declarar uma época concluída e iniciar a próxima época.
        #Você pode configurá-lo igual a samples / batch_size, que é uma escolha típica.
        steps_per_epoch=(training_set.n // training_set.batch_size),
        #A iteração é um processamento único para avançar e retroceder um lote de imagens (no nosso caso é 32, então 32 imagens são processadas em uma iteração).
        epochs=EPOCHS,
        validation_data = test_set,
        validation_steps= (test_set.n // test_set.batch_size),
        verbose=1,
        callbacks = [early_stopping_monitor]
      )

# atualizo valor da epoca caso o treinamento tenha finalizado antes do valor de epoca que foi iniciado
EPOCHS = len(classifier.history["loss"])

print("[INFO] Salvando modelo treinado ...")

#para todos arquivos ficarem com a mesma data e hora. Armazeno na variavel
file_date = getDateStr()
model.save('./models/'+FILE_NAME+file_date+'.h5')
print('[INFO] modelo: ./models/'+FILE_NAME+file_date+'.h5 salvo!')

end = time.time()

print("[INFO] Tempo de execução da CNN: %.1f min" %(getTimeMin(start,end)))

print('[INFO] Summary: ')
model.summary()

print("\n[INFO] Avaliando a CNN...")
score = model.evaluate(test_set, steps=(test_set.n // test_set.batch_size), verbose=1)
print('[INFO] Accuracy: %.2f%%' % (score[1]*100), '| Loss: %.5f' % (score[0]))

print("[INFO] Sumarizando loss e accuracy para os datasets 'train_net' e 'test_net'")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,EPOCHS), classifier.history["loss"], label="train_loss")
plt.plot(np.arange(0,EPOCHS), classifier.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,EPOCHS), classifier.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0,EPOCHS), classifier.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel(f"Epochs: {EPOCHS}")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('./models/graphics/'+FILE_NAME+file_date+'.png', bbox_inches='tight')

print('[INFO] Gerando imagem do modelo de camadas da CNN')
plot_model(model, to_file='./models/image/'+FILE_NAME+file_date+'.png', show_shapes = True)

print('[INFO] Gerando matriz de confusao')

# predições no conjunto de teste
Y_pred = model.predict(test_set)
# converter predições em classes
Y_pred_classes = np.argmax(Y_pred, axis=1)
# converter observações em classes
Y_true = test_set.classes
# calcular matriz de confusão
cm = confusion_matrix(Y_true, Y_pred_classes)

plt.figure(figsize=(10, 8))
#CLASSE=['A', 'B',  'C' , 'D', 'E','F', 'G', 'I', 'L','M', 'N', 'O','P','Q','R','S','T', 'U', 'V','W','Y']
#CLASSE=['0', '1',  '2' , '3', '4','5', '6', '7', '8','9']
CLASSE=['adulto', 'america',  'casa' , 'gasolina', 'juntos','lei', 'palavra', 'pedra', 'pequeno','verbo']
sns.heatmap(cm, annot=True, linewidth=.5, fmt='d')
# Plotando a matriz de confusão normalizada
#plt.imshow(cm_normalized, cmap=plt.cm.Blues, vmax=100, vmin=0)
plt.title('Matriz de Confusão')
plt.xlabel('Previsões')
plt.ylabel('Rótulos verdadeiros')
tick_marks = np.arange(len(CLASSE))
plt.xticks(tick_marks + 0.5, CLASSE)
plt.yticks(tick_marks + 0.5, CLASSE)


plt.show()
plt.savefig('./models'+FILE_NAME+file_date+'.png')


print('\n[INFO] [FIM]: ' + getDateStr())
print('\n\n')
