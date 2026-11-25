# Luna.cpp

Motor experimental de inferência local em C++, desenvolvido para executar modelos GGUF em CPU, inclusive em dispositivos Android com Termux. O objetivo é que o Luna tenha seu próprio motor de inferência, sem depender do llama.cpp como runtime.

> **Estado do repositório:** o código-fonte da versão mais recente ainda não foi publicado na branch principal. Esta branch não contém, no momento, uma versão que possa ser compilada ou instalada.

## Estado do desenvolvimento

- **Inferência textual:** uma cópia de desenvolvimento do Luna executou o modelo TinyStories GPT-2 em GGUF no Termux. Esse teste não comprova compatibilidade com todos os modelos GGUF.
- **Qwen3.5 texto:** o protótipo separado carregou Qwen3.5 0.8B real no Termux e respondeu corretamente a uma pergunta simples de aritmética, com geração medida em aproximadamente 1 token por segundo. Esse teste inicial não comprova qualidade geral nem compatibilidade com todos os modelos ou quantizações.
- **Visão:** um encoder experimental foi comparado parcialmente com uma referência numérica usando pesos reais de um modelo visual e dados de imagem sintéticos. Isso não demonstra compreensão de fotos.
- **Integração multimodal:** existe um protótipo que liga encoder e gerador, mas falta validar o encoder e demonstrar uma descrição de foto com os modelos reais juntos.

Esses resultados se referem a cópias de desenvolvimento externas a esta branch; não representam funcionalidades disponíveis para instalação nesta branch.

## Próximas etapas

1. Publicar o código-fonte correspondente à versão mais recente após verificar a integridade do envio.
2. Ampliar os testes do backend Qwen3.5 real em texto.
3. Validar o encoder visual e o fluxo completo de imagem até descrição.
4. Medir uso de memória e desempenho no Termux.
5. Documentar instalação, modelos compatíveis e limitações com base em testes reproduzíveis.

## Documentação anterior

O [relatório técnico](RELATORIO_LUNA_CPP.md) analisa um pacote anterior do Luna.cpp. Ele é histórico e não deve ser interpretado como descrição confirmada da versão mais recente.

## Aviso

O projeto é experimental e não está pronto para produção. Não há garantia de suporte a uma família de modelos apenas porque ela usa GGUF. Modelos e pesos não são distribuídos nesta branch.
