# Luna.cpp

Motor experimental de inferência local em C++, desenvolvido para executar modelos GGUF em CPU, inclusive em dispositivos Android com Termux. O objetivo é que o Luna tenha seu próprio motor de inferência, sem depender do llama.cpp como runtime.

> **Estado do repositório:** o código-fonte da versão mais recente ainda não foi publicado aqui. Este repositório não contém, no momento, uma versão que possa ser compilada ou instalada a partir da branch principal.

## Estado do desenvolvimento

- **Inferência textual:** uma cópia de desenvolvimento do Luna executou o modelo TinyStories GPT-2 em GGUF no Termux. Esse teste não comprova compatibilidade com todos os modelos GGUF.
- **Qwen3.5:** componentes do tokenizador e do gerador estão em desenvolvimento e tiveram testes isolados. A geração completa com Qwen3.5 ainda não foi demonstrada.
- **Visão:** um encoder experimental foi comparado com uma referência numérica usando pesos reais de um modelo visual e dados de imagem sintéticos. Isso não demonstra compreensão de fotos.
- **Integração multimodal:** ainda falta conectar e validar o modelo visual e o gerador para receber uma imagem e produzir uma descrição.

Esses resultados se referem a cópias de desenvolvimento externas a esta branch; não representam funcionalidades disponíveis para instalação neste repositório.

## Próximas etapas

1. Publicar o código-fonte correspondente à versão mais recente após recuperá-lo e verificá-lo.
2. Completar o backend Qwen3.5 e testar a geração com pesos reais.
3. Integrar o pré-processamento de imagens, o encoder visual e o gerador de texto.
4. Testar o fluxo completo, incluindo uso de memória e desempenho no Termux.
5. Documentar instalação, modelos compatíveis e limitações com base em testes reproduzíveis.

## Documentação anterior

O [relatório técnico](RELATORIO_LUNA_CPP.md) analisa um pacote anterior do Luna.cpp. Ele é histórico e não deve ser interpretado como descrição confirmada da versão mais recente.

## Aviso

O projeto é experimental e não está pronto para produção. Não há garantia de suporte a uma família de modelos apenas porque ela usa GGUF. Modelos e pesos não são distribuídos neste repositório.
