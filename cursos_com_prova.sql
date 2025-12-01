WITH ProdutoDeduplicado AS (
    -- 1. Identifica o ID do produto principal (o menor ID para cada nome) APENAS na instância 4661
    SELECT
        instancia_id,
        nome,
        MIN(id) AS produto_id_principal
    FROM
        produtos
    WHERE
        instancia_id = 4661 -- Filtra na CTE para otimização
        AND removido_em IS NULL
    GROUP BY
        instancia_id, nome
)
SELECT
    p_dedup.produto_id_principal AS produto_id,
    p_dedup.nome AS nome_produto,
    CASE
        WHEN COUNT(pr.id) > 0 THEN 'Sim'
        ELSE 'Não'
    END AS possui_prova,
    COUNT(pr.id) AS total_provas_cadastradas
FROM
    ProdutoDeduplicado p_dedup
LEFT JOIN
    provas pr ON p_dedup.produto_id_principal = pr.produto_id 
             AND p_dedup.instancia_id = pr.instancia_id
             AND pr.deletado_em IS NULL -- Considera apenas provas ativas
GROUP BY
    p_dedup.produto_id_principal, 
    p_dedup.nome
ORDER BY
    p_dedup.nome;