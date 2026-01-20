import pygame
import sys
import numpy as np
import random
from typing import Dict, Tuple, List, Optional

class QLearningAgent:
    
    def __init__(self, taxa_aprendizado=0.1, fator_desconto=0.9, epsilon=0.1):

        self.q_table = {}  # Tabela Q -> dicionario{estado: {ação: valor_q}}
        self.taxa_aprendizado = taxa_aprendizado
        self.fator_desconto = fator_desconto
        self.epsilon = epsilon
        
    def get_estado_key(self, tabuleiro: List[str]) -> str: # Transforma o tabuleiro em um chave
        return ''.join(tabuleiro)
    
    def get_acoes_validas(self, tabuleiro: List[str]) -> List[int]: # Retorna as posicoes vazias
        return [i for i, cell in enumerate(tabuleiro) if cell == ' ']
    
    def escolher_acao(self, tabuleiro: List[str], treinando=False) -> int:
        estado_key = self.get_estado_key(tabuleiro)
        acoes_validas = self.get_acoes_validas(tabuleiro)
        
        if not acoes_validas:
            return -1
        
        if estado_key not in self.q_table: # Se os estado ainda nao existe coloca na tabela
            self.q_table[estado_key] = {acao: 0.0 for acao in acoes_validas}
        
        if treinando and np.random.random() < self.epsilon: # Usando o epsilon-greedy
            # Exploracao -> escolhe uma acao aleatoria
            return np.random.choice(acoes_validas)
        else:
            # Explotacao: escolhe o maior valor Q
            q_values = {acao: self.q_table[estado_key].get(acao, 0.0) 
                       for acao in acoes_validas}
            return max(q_values, key=q_values.get)
    
    def atualizar_q_table(self, estado_atual: List[str], acao: int, recompensa: float, 
                          proximo_estado: List[str]):
        # Atualiza a Q-table usando Bellman
        
        estado_key = self.get_estado_key(estado_atual)
        proximo_estado_key = self.get_estado_key(proximo_estado)
        
        if estado_key not in self.q_table: # Se os estado ainda nao existe coloca na tabela
            acoes_validas = self.get_acoes_validas(estado_atual)
            self.q_table[estado_key] = {a: 0.0 for a in acoes_validas}
        
        if proximo_estado_key not in self.q_table:
            acoes_validas = self.get_acoes_validas(proximo_estado)
            self.q_table[proximo_estado_key] = {a: 0.0 for a in acoes_validas}
        
        
        q_atual = self.q_table[estado_key].get(acao, 0.0) # Valor  de Q atual para o par estado/ação
        
       
        if self.q_table[proximo_estado_key]:  # Melhor valor Q no proximo estado
            max_q_proximo = max(self.q_table[proximo_estado_key].values())
        else:
            max_q_proximo = 0.0
        
        # Eq Bellman
        novo_q = q_atual + self.taxa_aprendizado * (recompensa + self.fator_desconto * max_q_proximo - q_atual)
        
        # Atualiza o valor na Q-table
        self.q_table[estado_key][acao] = novo_q


class TreinadorIA:
    
    def __init__(self):

        self.ia_x = QLearningAgent(taxa_aprendizado=0.1, fator_desconto=0.9, epsilon=0.3)
        self.ia_o = QLearningAgent(taxa_aprendizado=0.1, fator_desconto=0.9, epsilon=0.3)
    
    def verificar_vitoria(self, tabuleiro: List[str]) -> Optional[str]:
        combinacoes = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Linhas 
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Colunas 
            [0, 4, 8], [2, 4, 6] # Diagonais
        ]
        
        for combo in combinacoes:
            if (tabuleiro[combo[0]] == tabuleiro[combo[1]] == 
                tabuleiro[combo[2]] != ' '):
                return tabuleiro[combo[0]]
        return None
    
    def tabuleiro_cheio(self, tabuleiro: List[str]) -> bool: # Verifica se o tabuleiro esta lotado
        return ' ' not in tabuleiro
    
    def get_recompensa(self, tabuleiro: List[str], jogador: str) -> float:
        vencedor = self.verificar_vitoria(tabuleiro)
        
        if vencedor == jogador:
            return 1.0  # Vitoria
        elif vencedor is not None:
            return -1.0  # Derrota
        elif self.tabuleiro_cheio(tabuleiro):
            return 0.01  # Empate
        else:
            return 0.0  # Jogo ainda nao terminou
    
    def simular_jogo(self) -> tuple:
        tabuleiro = [' '] * 9  # Tabuleiro vazio
        jogador_atual = 'X' # X comeca
        historico = [] # Registrar as jogadas
        
        # Enquanto o jogo nao acabou nem empatou
        while not self.verificar_vitoria(tabuleiro) and not self.tabuleiro_cheio(tabuleiro):
            estado_anterior = tabuleiro.copy()
            
            ia_atual = self.ia_x if jogador_atual == 'X' else self.ia_o 
            
            acao = ia_atual.escolher_acao(tabuleiro, treinando=True)
            
            if acao == -1: # Sem jogadas validas
                break
            
           
            tabuleiro[acao] = jogador_atual  # Faz a jogada
            
            historico.append({ # Salva jogada no historico
                'jogador': jogador_atual,
                'estado_anterior': estado_anterior,
                'acao': acao,
                'estado_atual': tabuleiro.copy()
            })

            jogador_atual = 'O' if jogador_atual == 'X' else 'X' # Muda para a proxima
        
        
        vencedor = self.verificar_vitoria(tabuleiro)
        
        for movimento in historico:
            jogador = movimento['jogador']
            ia_atual = self.ia_x if jogador == 'X' else self.ia_o
            
            # Calcula recompensa baseada no resultado final
            recompensa = self.get_recompensa(tabuleiro, jogador)
            
            # Atualiza a Q-table da IA
            ia_atual.atualizar_q_table(
                movimento['estado_anterior'],
                movimento['acao'],
                recompensa,
                movimento['estado_atual']
            )
        
        return vencedor, len(historico)
    
    def treinar(self, num_jogos: int = 10000) -> QLearningAgent:
        vitorias_x = 0
        vitorias_o = 0
        empates = 0
        
        for i in range(num_jogos):
            vencedor, num_movimentos = self.simular_jogo()
            
            # Contabiliza resultados
            if vencedor == 'X':
                vitorias_x += 1
            elif vencedor == 'O':
                vitorias_o += 1
            else:
                empates += 1
            
            # Reduz epsilon gradualmente para menos exploração
            if i % 1000 == 0:
                self.ia_x.epsilon = max(0.01, self.ia_x.epsilon * 0.99)
                self.ia_o.epsilon = max(0.01, self.ia_o.epsilon * 0.99)
        
        print(f"IA 'X' vitorias = {vitorias_x}, 'O' vitorias = {vitorias_o} e teve empates = {empates}")
        return self.ia_o


class JogoDaVelha:

    def __init__(self):

        pygame.init()
        
        self.LARGURA = 600
        self.ALTURA = 700
        self.TAMANHO_CELULA = 180
        self.MARGEM = 30
        

        self.BRANCO = (255, 255, 255)
        self.PRETO = (0, 0, 0)
        self.AZUL = (0, 100, 200)
        self.VERMELHO = (200, 0, 0)
        self.VERDE = (0, 150, 0)
        self.CINZA = (128, 128, 128)
        self.AZUL_CLARO = (173, 216, 230)


        self.tela = pygame.display.set_mode((self.LARGURA, self.ALTURA))
        pygame.display.set_caption("Jogo da Velha - Usando Q-table")
        self.clock = pygame.time.Clock()
        
        self.fonte = pygame.font.Font(None, 72)
        self.fonte_pequena = pygame.font.Font(None, 36)
        self.fonte_media = pygame.font.Font(None, 48)
        
    
        self.reset_jogo() # Inicializa o estado do jogo
        
        self.treinar_ia()
        
        # Tabela de desempenho
        self.vitorias_jogador = 0
        self.vitorias_ia = 0
        self.empates = 0
        
    def treinar_ia(self):
        treinador = TreinadorIA()
        self.ia = treinador.treinar(100000)
        
    def reset_jogo(self):
        self.tabuleiro = [' '] * 9
        self.jogador_atual = 'X'  
        self.jogo_terminado = False
        self.vencedor = None
        
    def get_posicao_mouse(self, pos: Tuple[int, int]) -> Optional[int]:
        x, y = pos
        
        if (self.MARGEM <= x <= self.LARGURA - self.MARGEM and # Ver se o clique foi na janela
            self.MARGEM + 50 <= y <= self.ALTURA - self.MARGEM - 50):
            
            
            col = (x - self.MARGEM) // self.TAMANHO_CELULA # Calcula qual foi clicada
            row = (y - self.MARGEM - 50) // self.TAMANHO_CELULA
            
            if 0 <= row < 3 and 0 <= col < 3:  # Converte para indice linear
                return row * 3 + col 
        
        return None
    
    def fazer_jogada(self, posicao: int, jogador: str) -> bool:
        if 0 <= posicao < 9 and self.tabuleiro[posicao] == ' ':
            self.tabuleiro[posicao] = jogador
            return True
        return False
    
    def verificar_vitoria(self) -> Optional[str]:
        combinacoes = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  
            [0, 4, 8], [2, 4, 6]   
        ]           
        
        for combo in combinacoes:
            if (self.tabuleiro[combo[0]] == self.tabuleiro[combo[1]] == 
                self.tabuleiro[combo[2]] != ' '):
                return self.tabuleiro[combo[0]]
        
        return None
    
    def tabuleiro_cheio(self) -> bool:
        return ' ' not in self.tabuleiro
    
    def jogada_ia(self):
        if not self.jogo_terminado:
            acao = self.ia.escolher_acao(self.tabuleiro, treinando=False)
            if acao != -1:
                self.fazer_jogada(acao, 'O')
                self.jogador_atual = 'X'
    
    def desenhar_tabuleiro(self):

        self.tela.fill(self.AZUL_CLARO)
        
        # Desenha o fundo branco do tabuleiro
        pygame.draw.rect(self.tela, self.BRANCO, 
                        (self.MARGEM, self.MARGEM + 50, 3 * self.TAMANHO_CELULA, 3 * self.TAMANHO_CELULA))
        
        # Desenha as linhas do grid
        for i in range(4):  # 4 linhas para fazer 3 células
            # Linhas horizontais
            y = self.MARGEM + 50 + i * self.TAMANHO_CELULA
            pygame.draw.line(self.tela, self.PRETO, 
                           (self.MARGEM, y), 
                           (self.LARGURA - self.MARGEM, y), 3)
            
            # Linhas verticais
            x = self.MARGEM + i * self.TAMANHO_CELULA
            pygame.draw.line(self.tela, self.PRETO, 
                           (x, self.MARGEM + 50), 
                           (x, self.ALTURA - self.MARGEM - 50), 3)
    
    def desenhar_simbolos(self):
        for i in range(9):
            if self.tabuleiro[i] != ' ':
                # Calcula posição da célula
                row = i // 3
                col = i % 3
                
                # Centro da célula
                x = self.MARGEM + col * self.TAMANHO_CELULA + self.TAMANHO_CELULA // 2
                y = self.MARGEM + 50 + row * self.TAMANHO_CELULA + self.TAMANHO_CELULA // 2
                
                if self.tabuleiro[i] == 'X':
                    # Desenha X com duas linhas diagonais
                    offset = 50
                    pygame.draw.line(self.tela, self.AZUL, 
                                   (x - offset, y - offset), 
                                   (x + offset, y + offset), 8)
                    pygame.draw.line(self.tela, self.AZUL, 
                                   (x + offset, y - offset), 
                                   (x - offset, y + offset), 8)
                else:  
                    pygame.draw.circle(self.tela, self.VERMELHO, (x, y), 50, 8)
    
    def desenhar_interface(self):
        titulo = self.fonte_media.render("Jogo da Velha - Usando Q-table", 
                                        True, self.PRETO)
        self.tela.blit(titulo, (self.LARGURA // 2 - titulo.get_width() // 2, 10))
        
        # Coloca as estatisticas
        stats_y = self.ALTURA - 80
        estatisticas = [
            f"Jogador: {self.vitorias_jogador}",
            f"IA: {self.vitorias_ia}",
            f"Empates: {self.empates}"
        ]
        
        for i, stat in enumerate(estatisticas):
            texto = self.fonte_pequena.render(stat, True, self.PRETO)
            x = 50 + i * 150
            self.tela.blit(texto, (x, stats_y))
        
        # Mensagens de status baseadas no estado do jogo
        if self.jogo_terminado:
            if self.vencedor == 'X':
                mensagem = "Fez o impossivel"
                cor = self.VERDE
            elif self.vencedor == 'O':
                mensagem = "IA ganhou!"
                cor = self.VERMELHO
            else:
                mensagem = "Empate! Boa"
                cor = self.CINZA
            
            # Exibe resultado
            texto = self.fonte_pequena.render(mensagem, True, cor)
            self.tela.blit(texto, (self.LARGURA // 2 - texto.get_width() // 2, 
                                 self.ALTURA - 40))
            

            instrucao = self.fonte_pequena.render("Pressione ESPAÇO para jogar novamente", 
                                                True, self.PRETO)
            self.tela.blit(instrucao, (self.LARGURA // 2 - instrucao.get_width() // 2, 
                                     self.ALTURA - 20))
        else:
            # Indica de quem  joga
            turno = "Sua vez (X)" if self.jogador_atual == 'X' else "Vez da IA (O)"
            texto = self.fonte_pequena.render(turno, True, self.PRETO)
            self.tela.blit(texto, (self.LARGURA // 2 - texto.get_width() // 2, 
                                 self.ALTURA - 40))
    
    def atualizar_estatisticas(self):

        if self.vencedor == 'X':
            self.vitorias_jogador += 1
        elif self.vencedor == 'O':
            self.vitorias_ia += 1
        else:
            self.empates += 1
    
    def processar_eventos(self):

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and self.jogo_terminado:
                    # Reinicia o jogo
                    self.reset_jogo()
    
            elif evento.type == pygame.MOUSEBUTTONDOWN:
                # Processa jogada do usuario
                if not self.jogo_terminado and self.jogador_atual == 'X':
                    posicao = self.get_posicao_mouse(evento.pos)
                    if posicao is not None:
                        if self.fazer_jogada(posicao, 'X'):
                            self.jogador_atual = 'O'
    
    def atualizar_jogo(self):
        if not self.jogo_terminado: # Ve se tem ganhador
            vencedor = self.verificar_vitoria()
            if vencedor:
                self.jogo_terminado = True
                self.vencedor = vencedor
                self.atualizar_estatisticas()
            elif self.tabuleiro_cheio(): # Empate
                self.jogo_terminado = True
                self.vencedor = None
                self.atualizar_estatisticas()
            
            elif self.jogador_atual == 'O': # Faz a jogada da ia
                pygame.time.wait(500)  
                self.jogada_ia()
    
    def executar(self):
        rodando = True
        
        while rodando:
            self.processar_eventos()
            self.atualizar_jogo()
            
            # Renderização
            self.desenhar_tabuleiro()
            self.desenhar_simbolos()
            self.desenhar_interface()
            
            pygame.display.flip()  # Atualiza a tela
            self.clock.tick(60)    

if __name__ == "__main__":
    try:
        jogo = JogoDaVelha()
        jogo.executar()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit()
    except Exception as e:
        print(f"Erro: {e}")
        pygame.quit()
        sys.exit()