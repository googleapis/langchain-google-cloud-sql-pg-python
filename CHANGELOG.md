# Changelog

## [0.8.0](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/compare/v0.7.0...v0.8.0) (2024-08-02)


### Features

* Add table name to default index name ([#171](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/171)) ([8e61bc7](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/8e61bc779bc8f803e40e76aaeffdb93c35a5c90f))
* Remove langchain-community dependency ([#172](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/172)) ([b4f40bb](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/b4f40bb389b40853e3deed37e1385a7866741231))


### Documentation

* Added vector store initialization from documents ([#174](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/issues/174)) ([eb2eac3](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/commit/eb2eac303f64e809e6f3fc9bc3307be163602a4e))

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
