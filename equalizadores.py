#!/usr/bin/env python
# coding: utf-8
"""
__author__: Henrique Matias
"""
 
# Importação das módulos necessarias
import numpy as np # Importa módulo para operações matemáticas
import matplotlib.pyplot as plt # Importa módulo pra plotagem de gráficos
import padasip as pd # Importa módulo para filtragem adaptativa
from scipy.signal import lfilter # Importa módulo para manipular filtragem
import warnings # Importa módulo para gerenciamento de avisos
warnings.filterwarnings('ignore') # Ignora avisos
 
# Criação de funções para auxilio na geração de sinais
def gera_sinal(numero_amostra):
   """ Gera um sinal aleatorio de acordo com o numero de amostras informado """
   return np.sign(np.random.randn(numero_amostra,1)) + 2*np.sign(np.random.randn(numero_amostra,1))
 
def cria_canal():
   """  Retorna um canal"""
   return [0.85, 0.5, 0.15]
 
def gera_ruido(numero_amostra):
   return 0.01*np.random.randn(numero_amostra,1)+np.random.randn(numero_amostra,1)
 
def convolucao(sinal, canal, numero_amostra):
   """ Realiza a convolucao entre o sinal e o canal """
   ac = np.ones((numero_amostra,1))
   for posicao_sinal in range (0, 2):
       if posicao_sinal == 0:
           ac[posicao_sinal] = sinal[posicao_sinal]*canal[posicao_sinal]
       else:
           for index_canal in range (0, len(canal) - 1):
               ac[posicao_sinal] += sinal[posicao_sinal] * canal[index_canal]
 
   for k in range (3, numero_amostra -1):
       a0 = sinal[k]
       a1 = sinal[k-1]
       a2 = sinal[k-2]
       ac[k] = a0 * canal[0] + a1 * canal[1] + a2 * canal[2]
 
   return ac
 
# Parametros para geração de sinal
numero_amostra = 500
canal = cria_canal()
 
# Geração do sinal
sinal = gera_sinal(numero_amostra)
ruido = gera_ruido(numero_amostra)
sinal_apos_canal = np.float64(convolucao(sinal, canal, numero_amostra) + ruido/10)
 
# Parametros para filtro LMS
numero_coeficientes_filtro_linear = 9
mi_passo_adaptacao = 0.01
tamanho_da_entrada = 1
iniciar_matriz_de_pesos = 'zeros'
 
# Parametros para filtro RLS
fator_de_esquecimento = 0.98
valor_de_inicializacao = 0.5
 
# Inicialização dos filtros
filter_lms = pd.filters.FilterLMS(tamanho_da_entrada, mi_passo_adaptacao, iniciar_matriz_de_pesos)
filter_rls = pd.filters.FilterRLS(tamanho_da_entrada, fator_de_esquecimento, valor_de_inicializacao)
 
# Realização da filtragem
sinal_lms, erro_lms, matriz_pesos_lms = filter_lms.run(sinal, sinal_apos_canal)
sinal_rls, erro_rls, matriz_pesos_rls = filter_rls.run(sinal, sinal_apos_canal)
 
# Geração de gráficos
# Gráfico do sinal
fig, ax = plt.subplots()
ax.plot(np.arange(0,numero_amostra), sinal, 'b.', label='Sinal original')
ax.plot(np.arange(0,numero_amostra),sinal_apos_canal, 'r.', label='Sinal após o canal')
ax.grid()
plt.title('Sinal')
plt.xlabel('Número de amostras')
plt.ylabel('Valor do sinal')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.show()
 
# Constelação do sinal
fig, ax = plt.subplots()
ax.plot(np.real(sinal_apos_canal),np.imag(sinal_apos_canal),'r.', label='Sinal após o canal')
ax.plot(np.real(sinal),np.imag(sinal), 'bo', label='Sinal original')
ax.grid()
plt.title('Constelação do sinal')
plt.xlabel('Valor do sinal')
ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
fig.show()
 
# Gráficos do Sinal Filtrado
# Sinal filtrado com LMS e adaptação do erro
plt.figure()
plt.plot(sinal_lms,'r.')
plt.grid()
plt.title('Sinal filtrado com LMS')
plt.xlabel('Número de amostras')
plt.ylabel('Valor do sinal')
plt.show()
plt.figure()
plt.plot(np.abs(erro_lms))
plt.grid()
plt.title('Valor médio do erro LMS')
plt.show()
 
# Sinal filtrado com RLS e adaptação do erro
plt.figure()
plt.plot(sinal_rls,'r.')
plt.grid()
plt.title('Sinal filtrado com RLS')
plt.xlabel('Número de amostras')
plt.ylabel('Valor do sinal')
plt.show()
plt.figure()
plt.plot(np.abs(erro_rls))
plt.grid()
plt.title('Valor médio do erro RLS')
plt.show()
 
# Comparativo entre os gráficos de filtragem
fig, ax = plt.subplots()
ax.plot(sinal_lms,'b.', label='Filtro LMS')
ax.plot(sinal_rls,'r.', label='Filtro RLS')
ax.grid()
plt.title('Comparativo entre os gráficos de filtragem')
plt.xlabel('Número de amostras')
plt.ylabel('Valor do sinal')
ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
fig.show()
 
# Comparativo entre os gráficos de adaptação do erro
fig, ax = plt.subplots()
ax.plot(np.abs(erro_rls),'r', label='Filtro RLS')
ax.plot(np.abs(erro_lms),'b', label='Filtro LMS')
ax.grid()
plt.title('Comparativo entre os gráficos de adaptação do erro')
plt.xlabel('Número de amostras')
plt.ylabel('Valor do sinal')
ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
fig.show()
