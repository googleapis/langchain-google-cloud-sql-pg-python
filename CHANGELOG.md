# Changelog

## [0.14.1](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/compare/v0.14.0...v0.14.1) (2025-07-11)


### Bug Fixes

* Add support for a metadata column named id ([#302](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/302)) ([ceffc44](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/ceffc449384288c695e5246e2553661dbb548fbf))

## [0.14.0](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/compare/v0.13.0...v0.14.0) (2025-04-29)


### Features

* Update Postgres VectorStore to expected LangChain functionality ([#290](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/290)) ([605c31d](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/605c31dad924d359e2de0bc26476c5830a0cba69))


### Bug Fixes

* **docs:** Fix link in README ([#293](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/293)) ([6bfc58c](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/6bfc58cd7d546cb454ff2a18c74cec287fc764cb))
* Update JSON conversion ([#296](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/296)) ([4313ba2](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/4313ba234f2328537ed9401c1152d78c1ab71440))

## [0.13.0](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/compare/v0.12.1...v0.13.0) (2025-03-17)


### Features

* **langgraph:** Add Langgraph Checkpointer ([#284](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/284)) ([14a4240](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/14a4240e1e5769b203e9463d016d08ac2e6f603e))


### Bug Fixes

* **deps:** Update dependency numpy to v2 ([#251](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/251)) ([a164aa2](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/a164aa2d54575461e993594a1c98b8fac0e06ea2))
* **engine:** Loop error on close ([#285](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/285)) ([e8bd4ae](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/e8bd4ae9f03a3e60af0a1335d976423b0ae6e41a))

## [0.12.1](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/compare/v0.12.0...v0.12.1) (2025-02-12)


### Bug Fixes

* Add write messages to Chat History ([#265](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/265)) ([0f69092](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/0f690921c95cfe1123f9a7bae6b88ce8748b4a34))
* Enquote column names to not match reserved keywords. ([#267](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/267)) ([ef63226](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/ef632262781453601364610e39aeac643c94efa1))
* Query and return only selected metadata columns ([#253](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/253)) ([a8cc5a2](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/a8cc5a20bb3bdcac1109b61ac0d86b341f2ae84d))

## [0.12.0](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/compare/v0.11.1...v0.12.0) (2025-01-06)


### Features

* Add engine_args argument to engine creation functions ([#242](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/242)) ([5f2f7b7](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/5f2f7b7754824fde23867137e32208ea276be43c))

## [0.11.1](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/compare/v0.11.0...v0.11.1) (2024-11-15)


### Documentation

* Add benefits to readme ([#224](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/224)) ([f791be0](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/f791be05ea32a0da473562a63f43dcd18c00d399))
* Adding a readme to itemise sample notebooks. ([#225](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/225)) ([4ce8be3](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/4ce8be3d43c5d0825859f21352131a71a62105af))

## [0.11.0](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/compare/v0.10.0...v0.11.0) (2024-10-04)


### Features

* Remove support for Python 3.8 ([#216](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/216)) ([1737adc](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/1737adc696e725488af860d031a22f6e6b66171b))

## [0.10.0](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/compare/v0.9.0...v0.10.0) (2024-09-17)


### ⚠ BREAKING CHANGES

* support async and sync versions of indexing methods
* remove _aexecute(), _execute(), _afetch(), and _fetch() methods

### Features

* Add from_engine_args method ([de16842](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/de168427f9884f33332086b68308e1225ee9e952))
* Add support for sync from_engine ([de16842](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/de168427f9884f33332086b68308e1225ee9e952))
* Allow non-uuid data types for vectorstore primary key ([#209](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/209)) ([ffaa87f](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/ffaa87fd864d1c3ffeb00a34370af9e986a37cf5))
* Refactor to support both async and sync usage ([de16842](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/de168427f9884f33332086b68308e1225ee9e952))


### Bug Fixes

* Replacing cosine_similarity and maximal_marginal_relevance local methods with the ones in langchain core. ([#190](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/190)) ([7f27092](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/7f2709225a1a5a71b33522dafd354dc7159c358f))
* Support async and sync versions of indexing methods ([de16842](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/de168427f9884f33332086b68308e1225ee9e952))
* Updating the minimum langchain core version to 0.2.36 ([#205](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/205)) ([0651231](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/0651231b7d77e0451ae769f78fe6dce3e724dec4))


### Documentation

* Update sample python notebooks to reflect the support for custom schema. ([#204](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/204)) ([7ef9335](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/7ef9335a45578273e9ffc0921f60a1c6cc3e89ed))


## [0.9.0](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/compare/v0.8.0...v0.9.0) (2024-09-05)


### Features

* Add support for custom schema names ([#191](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/191)) ([1e0566a](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/1e0566af98bf24c711315a791336ba212d240acd))

## [0.8.0](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/compare/v0.7.0...v0.8.0) (2024-09-04)


### Features

* Add table name to default index name ([#171](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/171)) ([8e61bc7](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/8e61bc779bc8f803e40e76aaeffdb93c35a5c90f))
* Remove langchain-community dependency ([#172](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/172)) ([b4f40bb](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/b4f40bb389b40853e3deed37e1385a7866741231))


### Bug Fixes

* Add caching for background loop/thread ([#184](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/184)) ([1489f81](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/1489f818c1d62bfee5c5a3bab42d380556662e82))
* Fix QueryOptions not applied to similarity search bug ([#185](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/185)) ([e5dca97](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/e5dca973d625c4df4c3e741a3ad8e95be0cd1472))
* Fixed extra char in requirements.txt ([#196](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/196)) ([50dc32f](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/50dc32f8ae476c98e3ed38a153096551ce02d340))


### Documentation

* Add index choosing guide ([#178](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/178)) ([e96ffb6](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/e96ffb6dc99425e4dafb8ac13730eed253e74c4e))
* Added vector store initialization from documents ([#174](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/174)) ([eb2eac3](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/eb2eac303f64e809e6f3fc9bc3307be163602a4e))
* Update README.md to fix 404 links to templates ([#182](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/182)) ([f10ae6c](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/f10ae6c9a8645874a5ab64e846ec540aeddf977a))

## [0.7.0](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/compare/v0.6.1...v0.7.0) (2024-07-23)


### Features

* Add similarity search score threshold select function ([#157](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/157)) ([71789f0](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/71789f06a9702ee2e037b084a88c1258b7232a4b))
* Added example for document saver ([#164](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/164)) ([13b909e](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/13b909e1fbc518728103ae6de0a1d8c462df8144))
* Auto-generate IDs upon adding embeddings ([#158](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/158)) ([a364514](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/a3645147f3d7fe0958d0420f948cf6afb8eb215b))
* Support IAM account override ([#160](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/160)) ([2de3cba](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/2de3cbae40d267a7b038a7b421999b5bb60c03d8))


### Bug Fixes

* Add key to engine constructor ([c12ded9](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/c12ded92abcb6a44e374f7b00afc1e17588e0688))
* Rename inner product distance search function to inner_product ([#168](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/168)) ([c5641c3](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/c5641c305e4c63d09f24c88dba679bcf1a4040b2))


### Documentation

* Add docstring to all methods ([#163](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/163)) ([61413f1](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/61413f10d9cb074a1fc82a742000827285208750))

## [0.6.1](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/compare/v0.6.0...v0.6.1) (2024-07-01)


### Bug Fixes

* Change IVFFlat `lists` default to 100 ([#149](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/149)) ([45854a1](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/45854a17929b07653acb641243db695f4cef9c7e))
* Use lazy refresh for Cloud SQL Connector ([#144](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/144)) ([cbae094](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/cbae09466ac6d7f96dba795316ec758287aaea58))


### Documentation

* Update API reference link ([#147](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/147)) ([7ce9f80](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/7ce9f800ac72d3e363d7704d0d65871195bd0bfb))

## [0.6.0](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/compare/v0.5.0...v0.6.0) (2024-06-18)


### Features

* Add support for quota project ([#137](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/137)) ([4f950b9](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/4f950b90b53b7e1943744ab95a00cd6b4045a892))

## [0.5.0](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/compare/v0.4.1...v0.5.0) (2024-05-30)


### Features

* Support LangChain v0.2 ([#133](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/133)) ([4364a04](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/4364a04bece61ab16f29cc37900dd79edc2128e2))

## [0.4.1](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/compare/v0.4.0...v0.4.1) (2024-05-02)


### Bug Fixes

* Missing quote in table name ([#123](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/123)) ([b490c81](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/b490c81ea96abca5a8acff1ca10f6ea8380a8ff1))
* Update required dep to SQLAlchemy[asyncio] ([#121](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/121)) ([d480760](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/d4807603d8573789199f2d277f59ef654e980198))


### Documentation

* Update readme codeblocks ([#124](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/124)) ([1c59d6b](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/1c59d6bf880f2a2755ef5a4f8eab8f7be0b95dc0))
* Update docs pipeline ([#109](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/109)) ([26c5342](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/26c534221dfb89a190bb1773b32493aae1ddf598))

## [0.4.0](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/compare/v0.3.0...v0.4.0) (2024-04-03)


### Features

* Support private ip ([#106](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/106)) ([0b8df1e](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/0b8df1ebd803d5d648396291788f6cede538a042))


### Bug Fixes

* API reference doc pipeline ([#108](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/108)) ([6ab1a40](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/6ab1a409193ba49a85879b02f0234990f22c249f))

## [0.3.0](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/compare/v0.2.0...v0.3.0) (2024-04-02)


### Features

* Add API reference docs ([#104](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/104)) ([43a9815](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/43a98157813ed40308f032ae85fb22962ca0311c))


### Bug Fixes

* Allow special characters in table name for vector store delete  ([#100](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/100)) ([7fc1c63](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/7fc1c635eee51864b70ad1fcfcec515cbf6ebea8))

## [0.2.0](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/compare/v0.1.0...v0.2.0) (2024-03-25)


### Features

* **ci:** Test against multiple versions ([#86](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/86)) ([a3f410d](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/a3f410d1bfda87aa3904d153140937c8e2a415f2))


### Bug Fixes

* Sql statement for non-nullable column ([#85](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/85)) ([e143e14](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/e143e14cc8ea12399be81c49f579a6c9872119ea))


### Documentation

* Add check for database ([#79](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/79)) ([4959ceb](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/4959ceb78aae27c8b5d48168ec096b8cd01b6e82))
* Add github links ([#84](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/84)) ([864f642](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/864f642c19b3409acffaea7c6479791b12dd059c))
* Update langchain_quick_start.ipynb ([#81](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/81)) ([5e79043](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/5e790436073b8c6e37be905a6215dc9ea5602adc))
* Update quickstart ([#76](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/76)) ([37b4380](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/37b4380871f50dc30274539d0f8a65664d023d35))

## 0.1.0 (2024-02-28)


### Features

* Add CloudSQL Postgresql chatmessagehistory with Integration Tests ([#23](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/23)) ([3ab9d4e](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/3ab9d4eeeb7fd99c4693ee697fb31a2ad9343872))
* Add indexing methods ([#21](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/21)) ([8eae440](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/8eae4406e41f234ef3c6a24621926c3f5c4555cb))
* Add PostgreSQL loader ([#49](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/49)) ([ada45ec](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/ada45ec3089254966e444d11c5c22f73b881d03b))
* Add PostgreSQLEngine ([#13](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/13)) ([b181f65](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/b181f658c2e769c74aefc6a53f587ca4a75682db))
* Add Vector Store ([#14](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/14)) ([f3e1127](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/f3e11276a69bf239d852e494eede37ed86b1b361))


### Documentation

* Add chatmessagehistory docs ([#48](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/48)) ([5f5df1d](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/5f5df1d0790dd0a90110a1c765a4f445c083267a))
* Add the code lab notebook ([#36](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/36)) ([ab7cbe4](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/ab7cbe4d0554a2a80a32e7feb7b4fc5c773ee379))
* Add vectorstore docs ([#22](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/22)) ([6c41df2](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/6c41df2f51c7b185d8d1b53ad6b12e42f32de224))
