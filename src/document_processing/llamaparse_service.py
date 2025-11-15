#!/usr/bin/env python3
"""
Enhanced LLaMA Cloud parsing with job ID monitoring and async processing
Integrated with RapidRFP RAG document processing pipeline
"""
import os
import asyncio
import time
import requests
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

from ..config.settings import Config
from .document_loader import ProcessedDocument, DocumentChunk

logger = logging.getLogger(__name__)

@dataclass
class LlamaParseResult:
    """Result from LlamaParse processing."""
    success: bool
    documents: List[Dict[str, Any]]
    job_id: Optional[str] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    parsing_method: str = "llamaparse"

class LlamaCloudParser:
    """Enhanced LLaMA Cloud parser with job monitoring integrated for RapidRFP RAG."""

    def __init__(self, api_key: str = None):
        """
        Initialize LlamaParse service.
        
        Args:
            api_key: LlamaParse API key (defaults to environment variable)
        """
        self.api_key = api_key or os.getenv("LLAMA_CLOUD_API_KEY", "llx-39hXdpTryilYqVOHcd3nu9FXaNRTIZS625IZNR11idRfqk1u")
        if not self.api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY not found in environment")
        
        self.base_url = "https://api.cloud.llamaindex.ai"
        
        # Default parsing settings
        self.default_settings = {
            'result_type': 'markdown',
            'language': 'en',
            'verbose': True,
            'num_workers': 1,
            'max_wait_time': 300,
            'poll_interval': 5
        }
        
        logger.info("LlamaParse service initialized")

    def parse_document_sync(self, file_path: str, **kwargs) -> LlamaParseResult:
        """Parse document synchronously using LlamaParse."""
        start_time = time.time()
        
        try:
            from llama_parse import LlamaParse

            # Merge settings
            settings = {**self.default_settings, **kwargs}

            parser = LlamaParse(
                api_key=self.api_key,
                result_type=settings.get("result_type", "markdown"),
                verbose=settings.get("verbose", True),
                language=settings.get("language", "en"),
                num_workers=settings.get("num_workers", 1)
            )

            logger.info(f"Parsing {file_path} with LlamaParse (sync)...")
            documents = parser.load_data(file_path)
            
            formatted_docs = [
                {
                    "text": doc.text, 
                    "metadata": {
                        **doc.metadata,
                        "source_file": file_path,
                        "parsing_method": "llamaparse_sync"
                    }
                } 
                for doc in documents
            ]

            processing_time = time.time() - start_time
            logger.info(f"Successfully parsed {len(formatted_docs)} document(s) in {processing_time:.2f}s")

            return LlamaParseResult(
                success=True,
                documents=formatted_docs,
                processing_time=processing_time,
                parsing_method="llamaparse_sync"
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Sync parsing failed for {file_path}: {str(e)}")
            return LlamaParseResult(
                success=False,
                documents=[],
                processing_time=processing_time,
                error_message=str(e)
            )

    async def parse_document_async(self, file_path: str, **kwargs) -> LlamaParseResult:
        """Parse document asynchronously using LlamaParse."""
        start_time = time.time()
        
        try:
            from llama_parse import LlamaParse
            import nest_asyncio

            # Enable nested async if needed
            try:
                nest_asyncio.apply()
            except RuntimeError:
                pass  # Already applied

            # Merge settings
            settings = {**self.default_settings, **kwargs}

            parser = LlamaParse(
                api_key=self.api_key,
                result_type=settings.get("result_type", "markdown"),
                verbose=settings.get("verbose", True),
                language=settings.get("language", "en"),
                num_workers=settings.get("num_workers", 1)
            )

            logger.info(f"Parsing {file_path} with LlamaParse (async)...")
            documents = await parser.aload_data(file_path)
            
            formatted_docs = [
                {
                    "text": doc.text, 
                    "metadata": {
                        **doc.metadata,
                        "source_file": file_path,
                        "parsing_method": "llamaparse_async"
                    }
                } 
                for doc in documents
            ]

            processing_time = time.time() - start_time
            logger.info(f"Successfully parsed {len(formatted_docs)} document(s) in {processing_time:.2f}s")

            return LlamaParseResult(
                success=True,
                documents=formatted_docs,
                processing_time=processing_time,
                parsing_method="llamaparse_async"
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Async parsing failed for {file_path}: {str(e)}")
            return LlamaParseResult(
                success=False,
                documents=[],
                processing_time=processing_time,
                error_message=str(e)
            )

    def submit_parsing_job(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Submit parsing job and return job ID for monitoring."""
        try:
            logger.info(f"Submitting parsing job for {file_path}...")

            with open(file_path, 'rb') as file:
                files = {'file': file}
                
                # Merge settings
                settings = {**self.default_settings, **kwargs}
                
                data = {
                    'result_type': settings.get('result_type', 'markdown'),
                    'language': settings.get('language', 'en'),
                    'verbose': str(settings.get('verbose', True)).lower()
                }

                headers = {
                    'Authorization': f'Bearer {self.api_key}'
                }

                response = requests.post(
                    f"{self.base_url}/api/v1/parsing/upload",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    job_id = result.get('id')
                    logger.info(f"Job submitted successfully. Job ID: {job_id}")
                    return {
                        "success": True,
                        "job_id": job_id,
                        "status": result.get('status', 'pending'),
                        "result": result
                    }
                else:
                    logger.error(f"Job submission failed: {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}"
                    }

        except Exception as e:
            logger.error(f"Job submission error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """Check parsing job status."""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}'
            }

            response = requests.get(
                f"{self.base_url}/api/v1/parsing/job/{job_id}",
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                status = result.get('status', 'unknown')
                logger.debug(f"Job {job_id} status: {status}")
                return {
                    "success": True,
                    "job_id": job_id,
                    "status": status,
                    "result": result
                }
            else:
                logger.error(f"Status check failed: {response.status_code}")
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }

        except Exception as e:
            logger.error(f"Status check error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """Get parsing job result."""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}'
            }

            response = requests.get(
                f"{self.base_url}/api/v1/parsing/job/{job_id}/result/markdown",
                headers=headers,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Retrieved result for job {job_id}")
                return {
                    "success": True,
                    "job_id": job_id,
                    "content": result,
                    "documents": self._format_documents(result, job_id)
                }
            else:
                logger.error(f"Result retrieval failed: {response.status_code}")
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }

        except Exception as e:
            logger.error(f"Result retrieval error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def wait_for_job_completion(self, job_id: str, max_wait_time: int = 300, poll_interval: int = 5) -> Dict[str, Any]:
        """Wait for job completion with polling."""
        start_time = time.time()

        logger.info(f"Waiting for job {job_id} to complete...")
        logger.info(f"Max wait time: {max_wait_time}s, Poll interval: {poll_interval}s")

        while time.time() - start_time < max_wait_time:
            status_result = self.check_job_status(job_id)

            if not status_result["success"]:
                return status_result

            status = status_result["status"]
            status_lower = status.lower()

            if status_lower == "success":
                logger.info(f"Job {job_id} completed successfully!")
                return self.get_job_result(job_id)
            elif status_lower in ["error", "failed"]:
                logger.error(f"Job {job_id} failed")
                return {
                    "success": False,
                    "error": f"Job failed with status: {status}",
                    "job_id": job_id
                }

            logger.info(f"Job still {status}, waiting {poll_interval}s...")
            time.sleep(poll_interval)

        logger.warning(f"Job {job_id} timed out after {max_wait_time}s")
        return {
            "success": False,
            "error": f"Job timed out after {max_wait_time}s",
            "job_id": job_id
        }

    def parse_with_job_monitoring(self, file_path: str, **kwargs) -> LlamaParseResult:
        """Complete workflow: submit job, monitor, and retrieve result."""
        start_time = time.time()
        
        # Merge settings
        settings = {**self.default_settings, **kwargs}
        
        # Submit job
        submit_result = self.submit_parsing_job(file_path, **settings)
        if not submit_result["success"]:
            return LlamaParseResult(
                success=False,
                documents=[],
                processing_time=time.time() - start_time,
                error_message=submit_result["error"]
            )

        job_id = submit_result["job_id"]

        # Wait for completion
        max_wait = settings.get("max_wait_time", 300)
        poll_interval = settings.get("poll_interval", 5)

        result = self.wait_for_job_completion(job_id, max_wait, poll_interval)
        processing_time = time.time() - start_time

        if result["success"]:
            logger.info(f"Successfully parsed {file_path}")
            logger.info(f"Found {len(result.get('documents', []))} document(s)")
            
            return LlamaParseResult(
                success=True,
                documents=result["documents"],
                job_id=job_id,
                processing_time=processing_time,
                parsing_method="llamaparse_job_monitoring"
            )
        else:
            return LlamaParseResult(
                success=False,
                documents=[],
                job_id=job_id,
                processing_time=processing_time,
                error_message=result["error"]
            )

    def _format_documents(self, raw_result: Dict, job_id: str = None) -> List[Dict]:
        """Format API result into document structure."""
        documents = []

        if isinstance(raw_result, dict):
            if "markdown" in raw_result:
                documents.append({
                    "text": raw_result["markdown"],
                    "metadata": {
                        "source": "llamaparse", 
                        "format": "markdown",
                        "job_id": job_id
                    }
                })
            elif "text" in raw_result:
                documents.append({
                    "text": raw_result["text"],
                    "metadata": {
                        "source": "llamaparse", 
                        "format": "text",
                        "job_id": job_id
                    }
                })
        elif isinstance(raw_result, list):
            for item in raw_result:
                if isinstance(item, dict):
                    documents.append({
                        "text": item.get("text", str(item)),
                        "metadata": {
                            **item.get("metadata", {}),
                            "source": "llamaparse",
                            "job_id": job_id
                        }
                    })

        return documents

    def convert_to_processed_document(self, 
                                    file_path: str, 
                                    parse_result: LlamaParseResult,
                                    chunk_size: int = None,
                                    chunk_overlap: int = None) -> Optional[ProcessedDocument]:
        """
        Convert LlamaParse result to ProcessedDocument format.
        
        Args:
            file_path: Original file path
            parse_result: Result from LlamaParse
            chunk_size: Chunk size for text splitting
            chunk_overlap: Overlap between chunks
            
        Returns:
            ProcessedDocument compatible with existing pipeline
        """
        if not parse_result.success:
            logger.error(f"Cannot convert failed parse result: {parse_result.error_message}")
            return None

        try:
            # Combine all document texts
            combined_text = "\n\n".join([doc["text"] for doc in parse_result.documents])
            
            if not combined_text.strip():
                logger.warning(f"No text content extracted from {file_path}")
                return None

            # Create document metadata
            metadata = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_extension': os.path.splitext(file_path)[1].lower(),
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                'total_chars': len(combined_text),
                'parsing_method': parse_result.parsing_method,
                'processing_time': parse_result.processing_time,
                'job_id': parse_result.job_id
            }

            # Create chunks using simple word-based splitting (compatible with existing pipeline)
            chunk_size = chunk_size or Config.CHUNK_SIZE
            chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
            
            chunks = self._create_chunks_simple(combined_text, metadata, chunk_size, chunk_overlap)
            total_tokens = sum(chunk.token_count for chunk in chunks)

            logger.info(f"Converted LlamaParse result: {len(chunks)} chunks, {total_tokens} tokens")

            return ProcessedDocument(
                chunks=chunks,
                metadata=metadata,
                total_tokens=total_tokens
            )

        except Exception as e:
            logger.error(f"Error converting LlamaParse result: {e}")
            return None

    def _create_chunks_simple(self, 
                            text: str, 
                            doc_metadata: Dict[str, Any],
                            chunk_size: int,
                            chunk_overlap: int) -> List[DocumentChunk]:
        """Create chunks using simple word-based splitting."""
        words = text.split()
        chunks = []
        chunk_index = 0
        start_word = 0

        while start_word < len(words):
            # Calculate end word for this chunk
            end_word = min(start_word + chunk_size, len(words))
            
            # Extract chunk words
            chunk_words = words[start_word:end_word]
            chunk_text = " ".join(chunk_words)
            
            # Estimate character positions
            start_char = len(" ".join(words[:start_word]))
            end_char = start_char + len(chunk_text)
            
            # Create chunk metadata
            chunk_metadata = {
                **doc_metadata,
                'chunk_index': chunk_index,
                'start_word': start_word,
                'end_word': end_word,
                'word_count': len(chunk_words)
            }
            
            chunks.append(DocumentChunk(
                content=chunk_text.strip(),
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                metadata=chunk_metadata,
                token_count=len(chunk_words)  # Approximate token count
            ))
            
            # Move to next chunk with overlap
            if end_word >= len(words):
                break
            
            start_word = end_word - chunk_overlap
            chunk_index += 1

        # Update total chunks in metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)

        return chunks

# Integration functions for the existing document processing pipeline
class EnhancedDocumentLoader:
    """Enhanced document loader with LlamaParse integration."""
    
    def __init__(self, 
                 chunk_size: int = None, 
                 chunk_overlap: int = None,
                 use_llamaparse: bool = True,
                 llamaparse_api_key: str = None):
        """
        Initialize enhanced document loader.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            use_llamaparse: Whether to use LlamaParse for supported formats
            llamaparse_api_key: LlamaParse API key
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.use_llamaparse = use_llamaparse
        
        # Initialize LlamaParse if available
        self.llamaparse = None
        if self.use_llamaparse:
            try:
                self.llamaparse = LlamaCloudParser(api_key=llamaparse_api_key)
                logger.info("LlamaParse integration enabled")
            except Exception as e:
                logger.warning(f"LlamaParse initialization failed: {e}")
                self.use_llamaparse = False

        # Fallback to original loader
        from .document_loader import DocumentLoader
        self.fallback_loader = DocumentLoader(chunk_size, chunk_overlap)

    def load_document(self, file_path: str, **kwargs) -> Optional[ProcessedDocument]:
        """
        Load document with LlamaParse integration.
        
        Args:
            file_path: Path to document file
            **kwargs: Additional arguments for LlamaParse
            
        Returns:
            ProcessedDocument or None if loading failed
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None

        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Determine if we should use LlamaParse
        llamaparse_formats = {'.pdf', '.docx', '.pptx', '.xlsx', '.html', '.xml'}
        use_llamaparse = (self.use_llamaparse and 
                         self.llamaparse and 
                         file_extension in llamaparse_formats)

        if use_llamaparse:
            logger.info(f"Using LlamaParse for {file_path}")
            return self._load_with_llamaparse(file_path, **kwargs)
        else:
            logger.info(f"Using fallback loader for {file_path}")
            return self.fallback_loader.load_document(file_path)

    def _load_with_llamaparse(self, file_path: str, **kwargs) -> Optional[ProcessedDocument]:
        """Load document using LlamaParse."""
        try:
            # Choose parsing method based on preferences
            parsing_method = kwargs.get('parsing_method', 'job_monitoring')
            
            if parsing_method == 'sync':
                parse_result = self.llamaparse.parse_document_sync(file_path, **kwargs)
            elif parsing_method == 'async':
                # Run async method in sync context
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    parse_result = loop.run_until_complete(
                        self.llamaparse.parse_document_async(file_path, **kwargs)
                    )
                except RuntimeError:
                    # No event loop running
                    parse_result = asyncio.run(
                        self.llamaparse.parse_document_async(file_path, **kwargs)
                    )
            else:  # job_monitoring (default)
                parse_result = self.llamaparse.parse_with_job_monitoring(file_path, **kwargs)

            if parse_result.success:
                # Convert to ProcessedDocument
                processed_doc = self.llamaparse.convert_to_processed_document(
                    file_path=file_path,
                    parse_result=parse_result,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                
                if processed_doc:
                    logger.info(f"Successfully loaded {file_path} with LlamaParse")
                    return processed_doc
                else:
                    logger.warning(f"LlamaParse succeeded but conversion failed for {file_path}")
                    
            else:
                logger.warning(f"LlamaParse failed for {file_path}: {parse_result.error_message}")

        except Exception as e:
            logger.error(f"Error using LlamaParse for {file_path}: {e}")

        # Fallback to original loader
        logger.info(f"Falling back to original loader for {file_path}")
        return self.fallback_loader.load_document(file_path)

    def estimate_processing_cost(self, file_path: str) -> Dict[str, Any]:
        """Estimate processing cost (delegates to fallback loader)."""
        return self.fallback_loader.estimate_processing_cost(file_path)