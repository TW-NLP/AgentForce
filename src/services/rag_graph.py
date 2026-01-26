"""
GraphRAG 核心实现 

"""

import faiss
import numpy as np
from openai import OpenAI
from typing import List, Dict, Optional
import networkx as nx
from collections import defaultdict
import tiktoken
from community import community_louvain
import json
import pickle
from pathlib import Path
from datetime import datetime
import hashlib
import uuid
import logging

# 文档解析库
from pypdf import PdfReader
import pdfplumber
from docx import Document as DocxDocument
import pandas as pd
import re
from config.settings import settings


logger = logging.getLogger(__name__)


class DocumentParser:
    """文档解析器：支持 PDF、DOCX、TXT、MD、CSV 等格式"""
    
    @staticmethod
    def parse_pdf(file_path: str) -> str:
        """解析 PDF 文件"""
        try:
            with pdfplumber.open(file_path) as pdf:
                text_parts = []
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"[Page {page_num}]\n{page_text}")
                    
                    tables = page.extract_tables()
                    for table_num, table in enumerate(tables, 1):
                        if table:
                            table_text = f"\n[Table {table_num} on Page {page_num}]\n"
                            for row in table:
                                table_text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
                            text_parts.append(table_text)
                
                return "\n\n".join(text_parts)
        except Exception as e:
            logger.warning(f"pdfplumber 解析失败，使用 pypdf: {e}")
            try:
                reader = PdfReader(file_path)
                text_parts = []
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text:
                        text_parts.append(f"[Page {page_num}]\n{text}")
                return "\n\n".join(text_parts)
            except Exception as e2:
                raise Exception(f"PDF 解析完全失败: {e2}")
    
    @staticmethod
    def parse_docx(file_path: str) -> str:
        """解析 DOCX 文件"""
        try:
            doc = DocxDocument(file_path)
            text_parts = []
            
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)
            
            for table_num, table in enumerate(doc.tables, 1):
                table_text = f"\n[Table {table_num}]\n"
                for row in table.rows:
                    row_text = " | ".join([cell.text for cell in row.cells])
                    table_text += row_text + "\n"
                text_parts.append(table_text)
            
            return "\n\n".join(text_parts)
        except Exception as e:
            raise Exception(f"DOCX 解析失败: {e}")
    
    @staticmethod
    def parse_txt(file_path: str) -> str:
        """解析纯文本文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def parse_markdown(file_path: str) -> str:
        """解析 Markdown 文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def parse_csv(file_path: str) -> str:
        """解析 CSV 文件"""
        try:
            df = pd.read_csv(file_path)
            text = f"CSV Data ({len(df)} rows x {len(df.columns)} columns)\n\n"
            text += df.to_string(index=False)
            return text
        except Exception as e:
            raise Exception(f"CSV 解析失败: {e}")
    
    @classmethod
    def parse_document(cls, file_path: str) -> str:
        """根据文件扩展名自动选择解析器"""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        parsers = {
            '.pdf': cls.parse_pdf,
            '.docx': cls.parse_docx,
            '.doc': cls.parse_docx,
            '.txt': cls.parse_txt,
            '.md': cls.parse_markdown,
            '.markdown': cls.parse_markdown,
            '.csv': cls.parse_csv,
        }
        
        parser = parsers.get(extension)
        if parser is None:
            raise ValueError(f"不支持的文件格式: {extension}")
        
        return parser(str(path))


class TextChunker:
    """智能文本分块器"""
    
    def __init__(self, chunk_size: int = 600, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """按句子分块（保持句子完整性）"""
        sentences = re.split(r'[.!?。！？]\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_length + sentence_words > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    s_length = len(s.split())
                    if overlap_length + s_length <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += s_length
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_words
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_text(self, text: str) -> List[str]:
        """分块文本"""
        return self.chunk_by_sentences(text)


class GraphRAGPipeline:
    """
    GraphRAG Pipeline
    
    1. 添加文档后自动保存
    2. 初始化时自动加载已有数据
    3. text_chunks 为空的问题
    """

    def __init__(self, llm_api_key: str, embedding_api_key: str, llm_url: str, 
                 embedding_url: str, embedding_name: str, embedding_dim: int,
                 llm_name: str, storage_dir: str = "./graphrag_storage"):
        
        self.llm_client = OpenAI(base_url=llm_url, api_key=llm_api_key)
        self.embedding_client = OpenAI(base_url=embedding_url, api_key=embedding_api_key)

        self.embedding_name = embedding_name
        self.llm_name = llm_name
        self.dimension = embedding_dim
        
        # 存储目录
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 文档管理
        self.document_parser = DocumentParser()
        self.text_chunker = TextChunker()
        self.documents = {}
        
        # UUID 映射
        self.uuid_to_docid = {}
        self.docid_to_uuid = {}
        
        # 图谱数据
        self.text_chunks = []
        self.chunk_to_doc = {}
        self.entities = {}
        self.relationships = []
        self.claims = []
        
        # 知识图谱
        self.graph = nx.Graph()
        
        # 社区结构
        self.communities = {}
        self.community_summaries = {}
        
        # FAISS 索引
        self.community_summary_index = None
        self.community_embeddings = []
        
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        self._auto_load_if_exists()
    
    def _auto_load_if_exists(self):
        """初始化时自动加载已有数据"""
        try:
            self.load("default")
            logger.info(f"✅ 自动加载知识库成功: {len(self.documents)} 个文档, {len(self.text_chunks)} 个chunks")
        except FileNotFoundError:
            logger.info("📝 未找到已有知识库，将创建新的")
        except Exception as e:
            logger.warning(f"⚠️ 加载知识库失败: {e}")
    
    # ==================== 文档管理 ====================
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def add_document(self, file_path: str, metadata: Optional[Dict] = None, 
                    doc_uuid: Optional[str] = None) -> str:
        """
        添加文档
        
        Returns:
            文档的 UUID
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if doc_uuid is None:
            doc_uuid = str(uuid.uuid4())
        
        file_hash = self._calculate_file_hash(str(file_path))
        
        # 检查文档是否已存在
        if file_hash in self.documents:
            logger.info(f"⚠️ 文档已存在: {file_path.name}")
            return self.docid_to_uuid.get(file_hash, doc_uuid)
        
        logger.info(f"📄 添加文档: {file_path.name} (UUID: {doc_uuid})")
        
        # 解析文档
        text = self.document_parser.parse_document(str(file_path))
        logger.info(f"  📝 文档解析完成，文本长度: {len(text)} 字符")
        
        # 分块
        chunks = self.text_chunker.chunk_text(text)
        logger.info(f"  ✂️ 分块完成: {len(chunks)} 个chunks")
        
        # 记录文档信息
        doc_info = {
            'uuid': doc_uuid,
            'path': str(file_path),
            'name': file_path.name,
            'hash': file_hash,
            'chunks': chunks,
            'chunk_ids': [],
            'metadata': metadata or {},
            'added_at': datetime.now().isoformat()
        }
        
        self.documents[file_hash] = doc_info
        self.uuid_to_docid[doc_uuid] = file_hash
        self.docid_to_uuid[file_hash] = doc_uuid
        
        # ★★★提取图元素并添加到 text_chunks ★★★
        chunk_start_id = len(self.text_chunks)
        logger.info(f"  🔍 开始提取图元素 (起始ID: {chunk_start_id})...")
        
        for chunk_id, chunk in enumerate(chunks):
            global_chunk_id = chunk_start_id + chunk_id
            
            # 提取图元素
            logger.info(f"    处理 chunk {chunk_id + 1}/{len(chunks)}...")
            elements = self.extract_graph_elements(chunk, global_chunk_id)
            
            # ★ 关键：添加到 text_chunks
            self.text_chunks.append(elements)
            self.chunk_to_doc[global_chunk_id] = file_hash
            doc_info['chunk_ids'].append(global_chunk_id)
            
            logger.info(f"      提取: {len(elements.get('entities', []))} 实体, "
                       f"{len(elements.get('relationships', []))} 关系")
        
        logger.info(f"  ✅ 完成: 提取了 {len(chunks)} 个文本块")
        logger.info(f"  📊 当前总计: {len(self.text_chunks)} 个chunks")
        
        # ★★★ 添加文档后自动保存 ★★★
        try:
            self.save("default")
            logger.info(f"  💾 知识库已自动保存")
        except Exception as e:
            logger.error(f"  ❌ 自动保存失败: {e}")
        
        return doc_uuid
    
    def remove_document(self, doc_id: str):
        """删除文档（支持 UUID 或内部 ID）"""
        if doc_id in self.uuid_to_docid:
            internal_doc_id = self.uuid_to_docid[doc_id]
            doc_uuid = doc_id
        elif doc_id in self.documents:
            internal_doc_id = doc_id
            doc_uuid = self.docid_to_uuid.get(doc_id)
        else:
            raise ValueError(f"文档不存在: {doc_id}")
        
        logger.info(f"🗑️ 删除文档: {self.documents[internal_doc_id]['name']}")
        
        # 标记删除的 chunks
        chunk_ids = set(self.documents[internal_doc_id]['chunk_ids'])
        for chunk_id in chunk_ids:
            if chunk_id < len(self.text_chunks):
                self.text_chunks[chunk_id] = {'entities': [], 'relationships': [], 'claims': []}
            self.chunk_to_doc.pop(chunk_id, None)
        
        # 删除映射
        if doc_uuid:
            self.uuid_to_docid.pop(doc_uuid, None)
            self.docid_to_uuid.pop(internal_doc_id, None)
        
        del self.documents[internal_doc_id]
        
        # 自动保存
        try:
            self.save("default")
            logger.info("  💾 删除后已自动保存")
        except Exception as e:
            logger.error(f"  ❌ 自动保存失败: {e}")
        
        logger.info("  ✅ 文档已删除")
    
    def list_documents(self) -> List[Dict]:
        """列出所有文档"""
        return [
            {
                'uuid': info['uuid'],
                'id': doc_id,
                'name': info['name'],
                'path': info['path'],
                'chunks': len(info['chunks']),
                'added_at': info['added_at'],
                'metadata': info['metadata']
            }
            for doc_id, info in self.documents.items()
        ]
    
    # ==================== 图元素提取 ====================
    
    def extract_graph_elements(self, text: str, chunk_id: int) -> Dict:
        """从文本提取图元素"""
        
        prompt = f"""从以下文本中提取结构化信息，返回JSON格式。

文本:
{text}

提取内容:
1. entities: [{{"name": "实体名", "type": "类型", "description": "描述"}}]
2. relationships: [{{"source": "源实体", "target": "目标实体", "description": "关系", "strength": 1-10}}]
3. claims: [{{"subject": "主体", "object": "客体", "type": "FACT/OPINION", "description": "描述", "date": "时间"}}]

只返回JSON，不要其他内容。
"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                messages=[
                    {"role": "system", "content": "你是知识图谱专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.debug(f"Chunk {chunk_id} 提取结果: {len(result.get('entities', []))} 实体")
            return result
            
        except Exception as e:
            logger.error(f"提取失败 (chunk {chunk_id}): {e}")
            return {"entities": [], "relationships": [], "claims": []}
    
    def summarize_entity(self, entity_name: str, descriptions: List[str]) -> str:
        """合并实体描述"""
        if len(descriptions) == 1:
            return descriptions[0]
        
        combined = "\n".join([f"- {desc}" for desc in descriptions])
        
        prompt = f"""整合以下关于"{entity_name}"的描述为一个摘要（150-200词）：

{combined}

只返回摘要。"""

        response = self.llm_client.chat.completions.create(
            model=self.llm_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300
        )
        
        return response.choices[0].message.content.strip()
    
    # ==================== 图谱构建 ====================
    
    def merge_entities_and_relationships(self):
        """合并实体和关系"""
        logger.info(f"📊 开始合并实体和关系 (text_chunks数量: {len(self.text_chunks)})...")
        
        self.entities = {}
        self.relationships = []
        
        entity_descriptions = defaultdict(list)
        entity_types = {}
        entity_sources = defaultdict(set)
        
        # ★★★  text_chunks 是否为空 ★★★
        if not self.text_chunks:
            logger.warning("⚠️ text_chunks 为空！请先添加文档。")
            return
        
        for chunk_id, chunk_data in enumerate(self.text_chunks):
            entities = chunk_data.get('entities', [])
            logger.debug(f"  Chunk {chunk_id}: {len(entities)} 实体")
            
            for entity in entities:
                name = entity['name']
                entity_descriptions[name].append(entity['description'])
                entity_types[name] = entity['type']
                entity_sources[name].add(chunk_id)
        
        logger.info(f"  发现 {len(entity_descriptions)} 个唯一实体")
        logger.info("  生成实体摘要...")
        
        for entity_name, descriptions in entity_descriptions.items():
            summary = self.summarize_entity(entity_name, descriptions)
            self.entities[entity_name] = {
                'description': summary,
                'type': entity_types[entity_name],
                'source_ids': list(entity_sources[entity_name])
            }
        
        # 合并关系
        relationship_map = defaultdict(lambda: {'descriptions': [], 'strengths': [], 'sources': set()})
        
        for chunk_id, chunk_data in enumerate(self.text_chunks):
            for rel in chunk_data.get('relationships', []):
                key = (rel['source'], rel['target'])
                relationship_map[key]['descriptions'].append(rel['description'])
                relationship_map[key]['strengths'].append(rel.get('strength', 5))
                relationship_map[key]['sources'].add(chunk_id)
        
        for (source, target), data in relationship_map.items():
            if source in self.entities and target in self.entities:
                self.relationships.append({
                    'source': source,
                    'target': target,
                    'description': '; '.join(data['descriptions']),
                    'weight': float(np.mean(data['strengths'])),
                    'source_ids': list(data['sources'])
                })
        
        logger.info(f"  ✅ 完成: {len(self.entities)} 实体, {len(self.relationships)} 关系")
    
    def build_graph(self):
        """构建知识图谱"""
        logger.info("🕸️ 构建知识图谱...")
        self.graph = nx.Graph()
        
        for entity_name, entity_data in self.entities.items():
            self.graph.add_node(
                entity_name,
                type=entity_data['type'],
                description=entity_data['description']
            )
        
        for rel in self.relationships:
            self.graph.add_edge(
                rel['source'],
                rel['target'],
                weight=rel['weight'],
                description=rel['description']
            )
        
        logger.info(f"  ✅ 图谱: {self.graph.number_of_nodes()} 节点, {self.graph.number_of_edges()} 边")
    
    def detect_hierarchical_communities(self, max_level: int = 3):
        """层次化社区检测"""
        logger.info("👥 社区检测...")
        
        self.communities = {}
        current_graph = self.graph.copy()
        
        for level in range(max_level):
            partition = community_louvain.best_partition(
                current_graph,
                weight='weight',
                resolution=1.0
            )
            
            communities_at_level = defaultdict(list)
            for node, comm_id in partition.items():
                communities_at_level[comm_id].append(node)
            
            self.communities[level] = dict(communities_at_level)
            logger.info(f"  Level {level}: {len(communities_at_level)} 个社区")
            
            if len(communities_at_level) <= 1:
                break
            
            # 构建下一层
            next_graph = nx.Graph()
            for comm_id in communities_at_level.keys():
                next_graph.add_node(f"comm_{level}_{comm_id}")
            
            for u, v, data in current_graph.edges(data=True):
                comm_u = partition[u]
                comm_v = partition[v]
                if comm_u != comm_v:
                    edge_key = (f"comm_{level}_{comm_u}", f"comm_{level}_{comm_v}")
                    if next_graph.has_edge(*edge_key):
                        next_graph[edge_key[0]][edge_key[1]]['weight'] += data.get('weight', 1)
                    else:
                        next_graph.add_edge(*edge_key, weight=data.get('weight', 1))
            
            current_graph = next_graph
    
    def generate_community_summary(self, level: int, community_id: int) -> str:
        """生成社区摘要"""
        nodes = self.communities[level][community_id]
        
        entities_info = []
        for node in nodes[:20]:
            if node in self.entities:
                entities_info.append(
                    f"- {node} ({self.entities[node]['type']}): "
                    f"{self.entities[node]['description'][:200]}"
                )
        
        relationships_info = []
        for rel in self.relationships:
            if rel['source'] in nodes and rel['target'] in nodes:
                relationships_info.append(
                    f"- {rel['source']} → {rel['target']}: {rel['description'][:150]}"
                )
        
        prompt = f"""生成社区摘要（300-400词）：

实体:
{chr(10).join(entities_info)}

关系:
{chr(10).join(relationships_info[:15])}

包括：主题、关键实体、关键发现、连接性。只返回摘要。"""

        response = self.llm_client.chat.completions.create(
            model=self.llm_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    def generate_all_community_summaries(self):
        """生成所有社区摘要"""
        logger.info("📝 生成社区摘要...")
        self.community_summaries = {}
        
        for level, communities in self.communities.items():
            for comm_id in communities.keys():
                summary = self.generate_community_summary(level, comm_id)
                self.community_summaries[(level, comm_id)] = summary
        
        logger.info(f"  ✅ 生成了 {len(self.community_summaries)} 个社区摘要")
    
    def build_community_summary_index(self):
        """构建向量索引"""
        logger.info("🔍 构建向量索引...")
        
        summaries = []
        summary_metadata = []
        
        for (level, comm_id), summary in self.community_summaries.items():
            summaries.append(summary)
            summary_metadata.append({
                'level': level,
                'community_id': comm_id,
                'summary': summary
            })
        
        if not summaries:
            logger.warning("⚠️ 没有社区摘要可索引")
            return
        
        # 生成 embeddings
        embeddings = []
        batch_size = 100
        
        logger.info(f"  生成 {len(summaries)} 个摘要的向量...")
        for i in range(0, len(summaries), batch_size):
            batch = summaries[i:i + batch_size]
            response = self.embedding_client.embeddings.create(
                model=self.embedding_name,
                input=batch
            )
            batch_embeddings = [np.array(item.embedding, dtype='float32') 
                              for item in response.data]
            embeddings.extend(batch_embeddings)
        
        self.community_embeddings = summary_metadata
        
        # 构建 FAISS
        embeddings_array = np.array(embeddings, dtype='float32')
        self.community_summary_index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(embeddings_array)
        self.community_summary_index.add(embeddings_array)
        
        logger.info(f"  ✅ 索引完成: {len(embeddings)} 个社区")
    
    # ==================== 索引构建 ====================
    
    def rebuild_index(self):
        """重建索引"""
        logger.info("=" * 60)
        logger.info("🔄 重建 GraphRAG 索引")
        logger.info("=" * 60)
        
        # ★★★ 检查是否有数据 ★★★
        if not self.text_chunks:
            logger.error("❌ text_chunks 为空！请先添加文档。")
            raise RuntimeError("没有文档可以索引，请先上传文档")
        
        logger.info(f"📊 数据统计: {len(self.text_chunks)} chunks, {len(self.documents)} 文档")
        
        logger.info("[1/5] 合并实体和关系...")
        self.merge_entities_and_relationships()
        logger.info(f"  完成: {len(self.entities)} 实体, {len(self.relationships)} 关系")
        
        if not self.entities:
            logger.error("❌ 没有提取到实体！请检查文档内容或 LLM 配置")
            raise RuntimeError("未能提取实体，索引构建失败")
        
        logger.info("[2/5] 构建知识图谱...")
        self.build_graph()
        logger.info(f"  完成: {self.graph.number_of_nodes()} 节点, {self.graph.number_of_edges()} 边")
        
        logger.info("[3/5] 社区检测...")
        self.detect_hierarchical_communities()
        
        logger.info("[4/5] 生成社区摘要...")
        self.generate_all_community_summaries()
        
        logger.info("[5/5] 构建向量索引...")
        self.build_community_summary_index()
        
        # ★★★重建索引后自动保存 ★★★
        try:
            self.save("default")
            logger.info("💾 索引重建后已自动保存")
        except Exception as e:
            logger.error(f"❌ 自动保存失败: {e}")
        
        logger.info("=" * 60)
        logger.info("✅ 索引重建完成!")
        logger.info("=" * 60)
    
    # ==================== 查询 ====================
    
    def global_query(self, question: str, top_k_communities: int = 10, return_sample=False) -> str:
        """查询知识库"""
        if self.community_summary_index is None:
            raise RuntimeError("索引未构建，请先上传文档并重建索引")
        
        # 检索社区
        query_embedding = self._get_embedding(question)
        query_embedding = np.array([query_embedding], dtype='float32')
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.community_summary_index.search(
            query_embedding, 
            min(top_k_communities, len(self.community_embeddings))
        )
        
        # 简单模式：直接返回社区摘要
        if getattr(settings, 'SIMPLE_RAG', False) or return_sample:
            search_results = []
            threshold = getattr(settings, 'T_SCORE', 0.5)
            
            for idx, score in zip(indices[0], scores[0]):
                if score >= threshold:
                    search_results.append(self.community_embeddings[idx]['summary'])
            
            if not search_results:
                return "抱歉，未找到相关信息。"
            
            end_str = ""
            for i, res in enumerate(search_results):
                end_str += f"社区摘要 {i+1}\n{res}\n\n"
            return end_str
        
        # Map-Reduce 模式
        community_answers = []
        
        for idx, score in zip(indices[0], scores[0]):
            if idx >= len(self.community_embeddings):
                continue
            
            comm_data = self.community_embeddings[idx]
            answer = self._ask_community(question, comm_data['summary'])
            
            if answer and len(answer.strip()) > 10:
                community_answers.append({
                    'level': comm_data['level'],
                    'community_id': comm_data['community_id'],
                    'content': answer,
                    'score': float(score)
                })
        
        return self._reduce_answers(question, community_answers)
    
    def _ask_community(self, question: str, community_summary: str) -> str:
        """询问单个社区"""
        prompt = f"""基于社区信息回答问题（2-3句话）。如果无关，回答"无相关信息"。

社区信息:
{community_summary}

问题: {question}

只返回答案。"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"社区查询失败: {e}")
            return ""
    
    def _reduce_answers(self, question: str, community_answers: List[Dict]) -> str:
        """综合答案"""
        if not community_answers:
            return "抱歉，在知识图谱中没有找到相关信息。"
        
        community_answers.sort(key=lambda x: x['score'], reverse=True)
        
        answers_text = []
        for i, ans_data in enumerate(community_answers[:10], 1):
            if ans_data['content'].lower() != "无相关信息":
                answers_text.append(f"{i}. {ans_data['content']}")
        
        if not answers_text:
            return "抱歉，找到的信息与问题不太相关。"
        
        combined = "\n".join(answers_text)
        
        prompt = f"""综合以下答案为一个连贯的最终答案（200-400词）：

问题: {question}

各社区答案:
{combined}

要求: 整合信息、消除冗余、保持清晰、呈现不同观点（如有）。

只返回最终答案。"""

        response = self.llm_client.chat.completions.create(
            model=self.llm_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600
        )
        
        return response.choices[0].message.content.strip()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本 embedding"""
        response = self.embedding_client.embeddings.create(
            model=self.embedding_name,
            input=text
        )
        return np.array(response.data[0].embedding, dtype='float32')
    
    # ==================== 持久化 ====================
    
    def save(self, name: str = "default"):
        """保存知识库"""
        save_dir = self.storage_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 保存知识库: {save_dir}")
        
        # 保存文档
        with open(save_dir / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        # 保存 UUID 映射
        with open(save_dir / "uuid_mappings.json", 'w', encoding='utf-8') as f:
            json.dump({
                'uuid_to_docid': self.uuid_to_docid,
                'docid_to_uuid': self.docid_to_uuid
            }, f, ensure_ascii=False, indent=2)
        
        # 保存图数据
        with open(save_dir / "graph_data.pkl", 'wb') as f:
            pickle.dump({
                'text_chunks': self.text_chunks,
                'chunk_to_doc': self.chunk_to_doc,
                'entities': self.entities,
                'relationships': self.relationships,
                'claims': self.claims,
                'communities': self.communities,
                'community_summaries': self.community_summaries,
                'community_embeddings': self.community_embeddings,
            }, f)
        
        # 保存图
        with open(save_dir / "graph.gpickle", 'wb') as f:
            pickle.dump(self.graph, f)
        
        # 保存 FAISS
        if self.community_summary_index:
            faiss.write_index(self.community_summary_index, 
                            str(save_dir / "faiss_index.bin"))
        
        logger.info(f"  ✅ 保存完成: {len(self.documents)} 文档, {len(self.text_chunks)} chunks")
    
    def load(self, name: str = "default"):
        """加载知识库"""
        load_dir = self.storage_dir / name
        
        if not load_dir.exists():
            raise FileNotFoundError(f"知识库不存在: {load_dir}")
        
        logger.info(f"📂 加载知识库: {load_dir}")
        
        # 加载文档
        with open(load_dir / "documents.json", 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        # 加载 UUID 映射
        uuid_path = load_dir / "uuid_mappings.json"
        if uuid_path.exists():
            with open(uuid_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
                self.uuid_to_docid = mappings['uuid_to_docid']
                self.docid_to_uuid = mappings['docid_to_uuid']
        
        # 加载图数据
        with open(load_dir / "graph_data.pkl", 'rb') as f:
            data = pickle.load(f)
            self.text_chunks = data['text_chunks']
            self.chunk_to_doc = data['chunk_to_doc']
            self.entities = data['entities']
            self.relationships = data['relationships']
            self.claims = data['claims']
            self.communities = data['communities']
            self.community_summaries = data['community_summaries']
            self.community_embeddings = data['community_embeddings']
        
        # 加载图
        with open(load_dir / "graph.gpickle", 'rb') as f:
            self.graph = pickle.load(f)
        
        # 加载 FAISS
        index_path = load_dir / "faiss_index.bin"
        if index_path.exists():
            self.community_summary_index = faiss.read_index(str(index_path))
        
        logger.info(f"  ✅ 加载完成: {len(self.documents)} 文档, "
                   f"{len(self.text_chunks)} chunks, {len(self.entities)} 实体")