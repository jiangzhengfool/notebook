> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/wjacketcn/article/details/50945887)

 从 3.1 开始，Spring 引入了对 Cache 的支持。其使用方法和原理都类似于 Spring 对事务管理的支持。Spring Cache 是作用在方法上的，其核心思想是这样的：当我们在调用一个缓存方法时会把该方法参数和返回结果作为一个键值对存放在缓存中，等到下次利用同样的参数来调用该方法时将不再执行该方法，而是直接从缓存中获取结果进行返回。所以在使用 Spring Cache 的时候我们要保证我们缓存的方法对于相同的方法参数要有相同的返回结果。

       使用 Spring Cache 需要我们做两方面的事：

n  声明某些方法使用缓存

n  配置 Spring 对 Cache 的支持

       和 Spring 对事务管理的支持一样，Spring 对 Cache 的支持也有基于注解和基于 XML 配置两种方式。下面我们先来看看基于注解的方式。

1       基于注解的支持
===============

       Spring 为我们提供了几个注解来支持 Spring Cache。其核心主要是 @Cacheable 和 @CacheEvict。使用 @Cacheable 标记的方法在执行后 Spring Cache 将缓存其返回结果，而使用 @CacheEvict 标记的方法会在方法执行前或者执行后移除 Spring Cache 中的某些元素。下面我们将来详细介绍一下 Spring 基于注解对 Cache 的支持所提供的几个注解。

1.1    @Cacheable
-----------------

       @Cacheable 可以标记在一个方法上，也可以标记在一个类上。当标记在一个方法上时表示该方法是支持缓存的，当标记在一个类上时则表示该类所有的方法都是支持缓存的。对于一个支持缓存的方法，Spring 会在其被调用后将其返回值缓存起来，以保证下次利用同样的参数来执行该方法时可以直接从缓存中获取结果，而不需要再次执行该方法。Spring 在缓存方法的返回值时是以键值对进行缓存的，值就是方法的返回结果，至于键的话，Spring 又支持两种策略，默认策略和自定义策略，这个稍后会进行说明。需要注意的是当一个支持缓存的方法在对象内部被调用时是不会触发缓存功能的。@Cacheable 可以指定三个属性，value、key 和 condition。

### 1.1.1  value 属性指定 Cache 名称

       value 属性是必须指定的，其表示当前方法的返回值是会被缓存在哪个 Cache 上的，对应 Cache 的名称。其可以是一个 Cache 也可以是多个 Cache，当需要指定多个 Cache 时其是一个数组。

   @Cacheable("cache1")//Cache 是发生在 cache1 上的

   public User find(Integer id) {

      returnnull;

   }

   @Cacheable({"cache1", "cache2"})//Cache 是发生在 cache1 和 cache2 上的

   public User find(Integer id) {

      returnnull;

   }

### 1.1.2  使用 key 属性自定义 key

       key 属性是用来指定 Spring 缓存方法的返回结果时对应的 key 的。该属性支持 SpringEL 表达式。当我们没有指定该属性时，Spring 将使用默认策略生成 key。我们这里先来看看自定义策略，至于默认策略会在后文单独介绍。

       自定义策略是指我们可以通过 Spring 的 EL 表达式来指定我们的 key。这里的 EL 表达式可以使用方法参数及它们对应的属性。使用方法参数时我们可以直接使用 “# 参数名” 或者“#p 参数 index”。下面是几个使用参数作为 key 的示例。

   @Cacheable(value="users", key="#id")

   public User find(Integer id) {

      returnnull;

   }

   @Cacheable(value="users", key="#p0")

   public User find(Integer id) {

      returnnull;

   }

   @Cacheable(value="users", key="#user.id")

   public User find(User user) {

      returnnull;

   }

   @Cacheable(value="users", key="#p0.id")

   public User find(User user) {

      returnnull;

   }

       除了上述使用方法参数作为 key 之外，Spring 还为我们提供了一个 root 对象可以用来生成 key。通过该 root 对象我们可以获取到以下信息。

| 

属性名称

 | 

描述

 | 

示例

 |
| 

methodName

 | 

当前方法名

 | 

#root.methodName

 |
| 

method

 | 

当前方法

 | 

#root.method.name

 |
| 

target

 | 

当前被调用的对象

 | 

#root.target

 |
| 

targetClass

 | 

当前被调用的对象的 class

 | 

#root.targetClass

 |
| 

args

 | 

当前方法参数组成的数组

 | 

#root.args[0]

 |
| 

caches

 | 

当前被调用的方法使用的 Cache

 | 

#root.caches[0].name

 |

       当我们要使用 root 对象的属性作为 key 时我们也可以将 “#root” 省略，因为 Spring 默认使用的就是 root 对象的属性。如：

   @Cacheable(value={"users", "xxx"}, key="caches[1].name")

   public User find(User user) {

      returnnull;

   }

### 1.1.3  condition 属性指定发生的条件

       有的时候我们可能并不希望缓存一个方法所有的返回结果。通过 condition 属性可以实现这一功能。condition 属性默认为空，表示将缓存所有的调用情形。其值是通过 SpringEL 表达式来指定的，当为 true 时表示进行缓存处理；当为 false 时表示不进行缓存处理，即每次调用该方法时该方法都会执行一次。如下示例表示只有当 user 的 id 为偶数时才会进行缓存。

   @Cacheable(value={"users"}, key="#user.id", condition="#user.id%2==0")

   public User find(User user) {

      System._out_.println("find user by user" + user);

      return user;

   }

1.2     @CachePut
-----------------

       在支持 Spring Cache 的环境下，对于使用 @Cacheable 标注的方法，Spring 在每次执行前都会检查 Cache 中是否存在相同 key 的缓存元素，如果存在就不再执行该方法，而是直接从缓存中获取结果进行返回，否则才会执行并将返回结果存入指定的缓存中。@CachePut 也可以声明一个方法支持缓存功能。与 @Cacheable 不同的是使用 @CachePut 标注的方法在执行前不会去检查缓存中是否存在之前执行过的结果，而是每次都会执行该方法，并将执行结果以键值对的形式存入指定的缓存中。

       @CachePut 也可以标注在类上和方法上。使用 @CachePut 时我们可以指定的属性跟 @Cacheable 是一样的。

   @CachePut("users")// 每次都会执行方法，并将结果存入指定的缓存中

   public User find(Integer id) {

      returnnull;

   }

1.3     @CacheEvict
-------------------

       @CacheEvict 是用来标注在需要清除缓存元素的方法或类上的。当标记在一个类上时表示其中所有的方法的执行都会触发缓存的清除操作。@CacheEvict 可以指定的属性有 value、key、condition、allEntries 和 beforeInvocation。其中 value、key 和 condition 的语义与 @Cacheable 对应的属性类似。即 value 表示清除操作是发生在哪些 Cache 上的（对应 Cache 的名称）；key 表示需要清除的是哪个 key，如未指定则会使用默认策略生成的 key；condition 表示清除操作发生的条件。下面我们来介绍一下新出现的两个属性 allEntries 和 beforeInvocation。

### 1.3.1  allEntries 属性

       allEntries 是 boolean 类型，表示是否需要清除缓存中的所有元素。默认为 false，表示不需要。当指定了 allEntries 为 true 时，Spring Cache 将忽略指定的 key。有的时候我们需要 Cache 一下清除所有的元素，这比一个一个清除元素更有效率。

   @CacheEvict(value="users", allEntries=true)

   public void delete(Integer id) {

      System._out_.println("delete user by id:" + id);

   }

### 1.3.2  beforeInvocation 属性

       清除操作默认是在对应方法成功执行之后触发的，即方法如果因为抛出异常而未能成功返回时也不会触发清除操作。使用 beforeInvocation 可以改变触发清除操作的时间，当我们指定该属性值为 true 时，Spring 会在调用该方法之前清除缓存中的指定元素。

   @CacheEvict(value="users", beforeInvocation=true)

   public void delete(Integer id) {

      System._out_.println("delete user by id:" + id);

   }

       其实除了使用 @CacheEvict 清除缓存元素外，当我们使用 Ehcache 作为实现时，我们也可以配置 Ehcache 自身的驱除策略，其是通过 Ehcache 的配置文件来指定的。由于 Ehcache 不是本文描述的重点，这里就不多赘述了，想了解更多关于 Ehcache 的信息，请查看我关于 Ehcache 的专栏。

1.4     @Caching
----------------

       @Caching 注解可以让我们在一个方法或者类上同时指定多个 Spring Cache 相关的注解。其拥有三个属性：cacheable、put 和 evict，分别用于指定 @Cacheable、@CachePut 和 @CacheEvict。

   @Caching(cacheable = @Cacheable("users"), evict = { @CacheEvict("cache2"),

         @CacheEvict(value = "cache3", allEntries = true) })

   public User find(Integer id) {

      returnnull;

   }

1.5     使用自定义注解
---------------

       Spring 允许我们在配置可缓存的方法时使用自定义的注解，前提是自定义的注解上必须使用对应的注解进行标注。如我们有如下这么一个使用 @Cacheable 进行标注的自定义注解。

@Target({ElementType._TYPE_, ElementType._METHOD_})

@Retention(RetentionPolicy._RUNTIME_)

@Cacheable(value="users")

public @interface MyCacheable {

}

       那么在我们需要缓存的方法上使用 @MyCacheable 进行标注也可以达到同样的效果。

   @MyCacheable

   public User findById(Integer id) {

      System._out_.println("find user by id:" + id);

      User user = new User();

      user.setId(id);

      user.setName("Name" + id);

      return user;

   }

2       配置 Spring 对 Cache 的支持
=============================

2.1     声明对 Cache 的支持
---------------------

### 2.1.1  基于注解

       配置 Spring 对基于注解的 Cache 的支持，首先我们需要在 Spring 的配置文件中引入 cache 命名空间，其次通过 <cache:annotation-driven /> 就可以启用 Spring 对基于注解的 Cache 的支持。

<?xml version=_"1.0"_ encoding=_"UTF-8"_?>

<beans xmlns=_"http://www.springframework.org/schema/beans"_

   xmlns:xsi=_"http://www.w3.org/2001/XMLSchema-instance"_

   xmlns:cache=_"http://www.springframework.org/schema/cache"_

   xsi:schemaLocation=_"http://www.springframework.org/schema/beans_

 _http://www.springframework.org/schema/beans/spring-beans-3.0.xsd_

 _http://www.springframework.org/schema/cache_

 _http://www.springframework.org/schema/cache/spring-cache.xsd"_>

   <cache:annotation-driven/>

</beans>

       <cache:annotation-driven/> 有一个 cache-manager 属性用来指定当前所使用的 CacheManager 对应的 bean 的名称，默认是 cacheManager，所以当我们的 CacheManager 的 id 为 cacheManager 时我们可以不指定该参数，否则就需要我们指定了。

       <cache:annotation-driven/> 还可以指定一个 mode 属性，可选值有 proxy 和 aspectj。默认是使用 proxy。当 mode 为 proxy 时，只有缓存方法在外部被调用的时候 Spring Cache 才会发生作用，这也就意味着如果一个缓存方法在其声明对象内部被调用时 Spring Cache 是不会发生作用的。而 mode 为 aspectj 时就不会有这种问题。另外使用 proxy 时，只有 public 方法上的 @Cacheable 等标注才会起作用，如果需要非 public 方法上的方法也可以使用 Spring Cache 时把 mode 设置为 aspectj。

       此外，<cache:annotation-driven/> 还可以指定一个 proxy-target-class 属性，表示是否要代理 class，默认为 false。我们前面提到的 @Cacheable、@cacheEvict 等也可以标注在接口上，这对于基于接口的代理来说是没有什么问题的，但是需要注意的是当我们设置 proxy-target-class 为 true 或者 mode 为 aspectj 时，是直接基于 class 进行操作的，定义在接口上的 @Cacheable 等 Cache 注解不会被识别到，那对应的 Spring Cache 也不会起作用了。

       需要注意的是 <cache:annotation-driven/> 只会去寻找定义在同一个 ApplicationContext 下的 @Cacheable 等缓存注解。

### 2.1.2  基于 XML 配置

       除了使用注解来声明对 Cache 的支持外，Spring 还支持使用 XML 来声明对 Cache 的支持。这主要是通过类似于 aop:advice 的 cache:advice 来进行的。在 cache 命名空间下定义了一个 cache:advice 元素用来定义一个对于 Cache 的 advice。其需要指定一个 cache-manager 属性，默认为 cacheManager。cache:advice 下面可以指定多个 cache:caching 元素，其有点类似于使用注解时的 @Caching 注解。cache:caching 元素下又可以指定 cache:cacheable、cache:cache-put 和 cache:cache-evict 元素，它们类似于使用注解时的 @Cacheable、@CachePut 和 @CacheEvict。下面来看一个示例：

   <cache:advice id=_"cacheAdvice"_ cache-manager=_"cacheManager"_>

      <cache:caching cache=_"users"_>

         <cache:cacheable method=_"findById"_ key=_"#p0"_/>

         <cache:cacheable method=_"find"_ key=_"#user.id"_/>

         <cache:cache-evict method=_"deleteAll"_ all-entries=_"true"_/>

      </cache:caching>

   </cache:advice>

       上面配置定义了一个名为 cacheAdvice 的 cache:advice，其中指定了将缓存 findById 方法和 find 方法到名为 users 的缓存中。这里的方法还可以使用通配符 “*”，比如“find*” 表示任何以 “find” 开始的方法。

       有了 cache:advice 之后，我们还需要引入 aop 命名空间，然后通过 aop:config 指定定义好的 cacheAdvice 要应用在哪些 pointcut 上。如：

   <aop:config proxy-target-class=_"false"_>

      <aop:advisor advice-ref=_"cacheAdvice"_ pointcut=_"execution(* com.xxx.UserService.*(..))"_/>

   </aop:config>

       上面的配置表示在调用 com.xxx.UserService 中任意公共方法时将使用 cacheAdvice 对应的 cache:advice 来进行 Spring Cache 处理。更多关于 Spring Aop 的内容不在本文讨论范畴内。

2.2     配置 CacheManager
-----------------------

       CacheManager 是 Spring 定义的一个用来管理 Cache 的接口。Spring 自身已经为我们提供了两种 CacheManager 的实现，一种是基于 Java API 的 ConcurrentMap，另一种是基于第三方 Cache 实现——Ehcache，如果我们需要使用其它类型的缓存时，我们可以自己来实现 Spring 的 CacheManager 接口或 AbstractCacheManager 抽象类。下面分别来看看 Spring 已经为我们实现好了的两种 CacheManager 的配置示例。

### 2.2.1  基于 ConcurrentMap 的配置

   <bean id=_"cacheManager"_ class=_"org.springframework.cache.support.SimpleCacheManager"_>

      <property name=_"caches"_>

         <set>

            <bean class=_"org.springframework.cache.concurrent.ConcurrentMapCacheFactoryBean"_ p:name=_"xxx"_/>

         </set>

      </property>

   </bean>

       上面的配置使用的是一个 SimpleCacheManager，其中包含一个名为 “xxx” 的 ConcurrentMapCache。

### 2.2.2  基于 Ehcache 的配置

   <!-- Ehcache 实现 -->

   <bean id=_"cacheManager"_ class=_"org.springframework.cache.ehcache.EhCacheCacheManager"_ p:cache-manager-ref=_"ehcacheManager"_/>

   <bean id=_"ehcacheManager"_ class=_"org.springframework.cache.ehcache.EhCacheManagerFactoryBean"_ p:config-location=_"ehcache-spring.xml"_/>

       上面的配置使用了一个 Spring 提供的 EhCacheCacheManager 来生成一个 Spring 的 CacheManager，其接收一个 Ehcache 的 CacheManager，因为真正用来存入缓存数据的还是 Ehcache。Ehcache 的 CacheManager 是通过 Spring 提供的 EhCacheManagerFactoryBean 来生成的，其可以通过指定 ehcache 的配置文件位置来生成一个 Ehcache 的 CacheManager。若未指定则将按照 Ehcache 的默认规则取 classpath 根路径下的 ehcache.xml 文件，若该文件也不存在，则获取 Ehcache 对应 jar 包中的 ehcache-failsafe.xml 文件作为配置文件。更多关于 Ehcache 的内容这里就不多说了，它不属于本文讨论的内容，欲了解更多关于 Ehcache 的内容可以参考我之前发布的 Ehcache 系列文章，也可以参考官方文档等。

3       键的生成策略
==============

       键的生成策略有两种，一种是默认策略，一种是自定义策略。

3.1     默认策略
------------

       默认的 key 生成策略是通过 KeyGenerator 生成的，其默认策略如下：

n  如果方法没有参数，则使用 0 作为 key。

n  如果只有一个参数的话则使用该参数作为 key。

n  如果参数多余一个的话则使用所有参数的 hashCode 作为 key。

       如果我们需要指定自己的默认策略的话，那么我们可以实现自己的 KeyGenerator，然后指定我们的 Spring Cache 使用的 KeyGenerator 为我们自己定义的 KeyGenerator。

       使用基于注解的配置时是通过 cache:annotation-driven 指定的.

   <cache:annotation-driven key-generator=_"userKeyGenerator"_/>

   <bean id=_"userKeyGenerator"_ class=_"com.xxx.cache.UserKeyGenerator"_/>

       而使用基于 XML 配置时是通过 cache:advice 来指定的。

   <cache:advice id=_"cacheAdvice"_ cache-manager=_"cacheManager"_ key-generator=_"userKeyGenerator"_>

   </cache:advice>

       需要注意的是此时我们所有的 Cache 使用的 Key 的默认生成策略都是同一个 KeyGenerator。

3.2     自定义策略
-------------

       自定义策略是指我们可以通过 Spring 的 EL 表达式来指定我们的 key。这里的 EL 表达式可以使用方法参数及它们对应的属性。使用方法参数时我们可以直接使用 “# 参数名” 或者“#p 参数 index”。下面是几个使用参数作为 key 的示例。

   @Cacheable(value="users", key="#id")

   public User find(Integer id) {

      returnnull;

   }

   @Cacheable(value="users", key="#p0")

   public User find(Integer id) {

      returnnull;

   }

   @Cacheable(value="users", key="#user.id")

   public User find(User user) {

      returnnull;

   }

   @Cacheable(value="users", key="#p0.id")

   public User find(User user) {

      returnnull;

   }

       除了上述使用方法参数作为 key 之外，Spring 还为我们提供了一个 root 对象可以用来生成 key。通过该 root 对象我们可以获取到以下信息。

| 

属性名称

 | 

描述

 | 

示例

 |
| 

methodName

 | 

当前方法名

 | 

#root.methodName

 |
| 

method

 | 

当前方法

 | 

#root.method.name

 |
| 

target

 | 

当前被调用的对象

 | 

#root.target

 |
| 

targetClass

 | 

当前被调用的对象的 class

 | 

#root.targetClass

 |
| 

args

 | 

当前方法参数组成的数组

 | 

#root.args[0]

 |
| 

caches

 | 

当前被调用的方法使用的 Cache

 | 

#root.caches[0].name

 |

       当我们要使用 root 对象的属性作为 key 时我们也可以将 “#root” 省略，因为 Spring 默认使用的就是 root 对象的属性。如：

   @Cacheable(value={"users", "xxx"}, key="caches[1].name")

   public User find(User user) {

      returnnull;

   }

4       Spring 单独使用 Ehcache
===========================

       前面介绍的内容是 Spring 内置的对 Cache 的支持，其实我们也可以通过 Spring 自己单独的使用 Ehcache 的 CacheManager 或 Ehcache 对象。通过在 Application Context 中配置 EhCacheManagerFactoryBean 和 EhCacheFactoryBean，我们就可以把对应的 EhCache 的 CacheManager 和 Ehcache 对象注入到其它的 Spring bean 对象中进行使用。

4.1     EhCacheManagerFactoryBean
---------------------------------

     EhCacheManagerFactoryBean 是 Spring 内置的一个可以产生 Ehcache 的 CacheManager 对象的 FactoryBean。其可以通过属性 configLocation 指定用于创建 CacheManager 的 Ehcache 配置文件的路径，通常是 ehcache.xml 文件的路径。如果没有指定 configLocation，则将使用默认位置的配置文件创建 CacheManager，这是属于 Ehcache 自身的逻辑，即如果在 classpath 根路径下存在 ehcache.xml 文件，则直接使用该文件作为 Ehcache 的配置文件，否则将使用 ehcache-xxx.jar 中的 ehcache-failsafe.xml 文件作为配置文件来创建 Ehcache 的 CacheManager。此外，如果不希望创建的 CacheManager 使用默认的名称（在 ehcache.xml 文件中定义的，或者是由 CacheManager 内部定义的），则可以通过 cacheManagerName 属性进行指定。下面是一个配置 EhCacheManagerFactoryBean 的示例。

   <!-- 定义 CacheManager -->

   <bean id=_"cacheManager"_ class=_"org.springframework.cache.ehcache.EhCacheManagerFactoryBean"_>

      <!-- 指定配置文件的位置 -->

      <property name=_"configLocation"_ value=_"/WEB-INF/config/ehcache.xml"_/>

      <!-- 指定新建的 CacheManager 的名称 -->

      <property name=_"cacheManagerName"_ value=_"cacheManagerName"_/>

   </bean>

4.2     EhCacheFactoryBean
--------------------------

       EhCacheFactoryBean 是用来产生 Ehcache 的 Ehcache 对象的 FactoryBean。定义 EhcacheFactoryBean 时有两个很重要的属性我们可以来指定。一个是 cacheManager 属性，其可以指定将用来获取或创建 Ehcache 的 CacheManager 对象，若未指定则将通过 CacheManager.create() 获取或创建默认的 CacheManager。另一个重要属性是 cacheName，其表示当前 EhCacheFactoryBean 对应的是 CacheManager 中的哪一个 Ehcache 对象，若未指定默认使用 beanName 作为 cacheName。若 CacheManager 中不存在对应 cacheName 的 Ehcache 对象，则将使用 CacheManager 创建一个名为 cacheName 的 Cache 对象。此外我们还可以通过 EhCacheFactoryBean 的 timeToIdle、timeToLive 等属性指定要创建的 Cache 的对应属性，注意这些属性只对 CacheManager 中不存在对应 Cache 时新建的 Cache 才起作用，对已经存在的 Cache 将不起作用，更多属性设置请参考 Spring 的 API 文档。此外还有几个属性是对不管是已经存在还是新创建的 Cache 都起作用的属性：statisticsEnabled、sampledStatisticsEnabled、disabled、blocking 和 cacheEventListeners，其中前四个默认都是 false，最后一个表示为当前 Cache 指定 CacheEventListener。下面是一个定义 EhCacheFactoryBean 的示例。

   <!-- 定义 CacheManager -->

   <bean id=_"cacheManager"_ class=_"org.springframework.cache.ehcache.EhCacheManagerFactoryBean"_>

      <!-- 指定配置文件的位置 -->

      <property name=_"configLocation"_ value=_"/WEB-INF/config/ehcache.xml"_/>

      <!-- 指定新建的 CacheManager 的名称 -->

      <property name=_"cacheManagerName"_ value=_"cacheManagerName"_/>

   </bean>

   <!-- 定义一个 Ehcache -->

   <bean id=_"userCache"_ class=_"org.springframework.cache.ehcache.EhCacheFactoryBean"_>

      <property name=_"cacheName"_ value=_"user"_/>

      <property name=_"cacheManager"_ ref=_"cacheManager"_/>

   </bean>